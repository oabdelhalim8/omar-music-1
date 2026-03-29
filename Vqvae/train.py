import os
import numpy as np
import torch
from utils.dataload import MidiDatasetVqvae, load_maestro_data, print_logger, load_midi_data
from torch.utils.data import DataLoader, Subset
from model.vqvae import VectorQuantizedVAE
from utils.midi_transfer import midi2pianoroll, pianoroll2midi
from utils.checkpoint_manager import CheckpointManager, DivergenceRiskTracker, RISK_POLICY, RiskLevel
from types import SimpleNamespace
import yaml
from tqdm import tqdm


class Lightning:
    def __init__(self, config_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(config_path,  'r') as fid:
            cfg = yaml.safe_load(fid)
            self.args = SimpleNamespace(**cfg)

        self.save_path = 'runs/checkpoints-{}'.format(self.args.model_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.logs = print_logger(self.save_path)
        self.generator_data()
        self.build_model()
        self.logs.info(cfg)

        # ── Metric-aware checkpoint retention ────────────────────────────
        ckpt_cfg = getattr(self.args, 'checkpoint', {})
        self.ckpt_manager = CheckpointManager(
            save_dir=self.save_path,
            model_name=self.args.model_name,
            save_top_k=ckpt_cfg.get('save_top_k', 5),
            keep_last_k=ckpt_cfg.get('keep_last_k', 3),
            weights_only=ckpt_cfg.get('weights_only', True),
            mode=ckpt_cfg.get('mode', 'min'),
        )

        # ── Divergence risk tracker ───────────────────────────────────────
        div_cfg = getattr(self.args, 'divergence', {})
        self.risk_tracker = DivergenceRiskTracker(
            gap_threshold=div_cfg.get('gap_threshold', 0.05),
            high_variance_threshold=div_cfg.get('high_variance_threshold', 0.02),
            history_window=div_cfg.get('history_window', 5),
        )
        self._cv_folds = div_cfg.get('cv_folds', 5)

    def generator_data(self):
        trn_data, val_data, self.vis_data = eval(self.args.data_func['func'])(**self.args.data_func['params'])
        self.iterTrain = DataLoader(MidiDatasetVqvae(trn_data, self.args.shape), batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
        self.iterValid = DataLoader(MidiDatasetVqvae(val_data, self.args.shape), batch_size=self.args.batch_size, pin_memory=True)

    def build_model(self):
        self.model = VectorQuantizedVAE(**self.args.net_params).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=False)

    def compute_accr(self, x, x_recon):
        x = x.detach().cpu().numpy()
        x_recon = x_recon.sigmoid().detach().round().cpu().numpy()

        equal = x == x_recon
        TP = equal[x==1].mean()
        FP = equal[x==0].mean()

        return TP, FP

    def compute_loss(self, x_recon, x):
        loss = self.criterion(x_recon, x)
        weight = ((x == 1) * 9 + 1).float()
        loss = (weight*loss).mean()

        # loss = self.criterion(x_recon, x).mean()

        return loss

    def train(self):
        self.model.train()

        loss, accr_TP, accr_FP = [], [], []
        for batch in tqdm(self.iterTrain):
            x = batch.to(self.device)
            q_loss, x_recon, perplexity = self.model(x)
            recon_loss = self.compute_loss(x_recon, x)
            _loss = recon_loss + q_loss

            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()

            _accr_TP, _accr_FP = self.compute_accr(x, x_recon)
            loss.append(recon_loss.item())
            accr_TP.append(_accr_TP)
            accr_FP.append(_accr_FP)

        return np.mean(loss), np.mean(accr_TP), np.mean(accr_FP)

    def valid(self):
        self.model.eval()

        with torch.no_grad():
            loss, accr_TP, accr_FP = [], [], []
            for batch in tqdm(self.iterValid):
                x = batch.to(self.device)
                _, x_recon, _ = self.model(x)
                recon_loss = self.compute_loss(x_recon, x)


                _accr_TP, _accr_FP = self.compute_accr(x, x_recon)
                loss.append(recon_loss.item())
                accr_TP.append(_accr_TP)
                accr_FP.append(_accr_FP)

        return np.mean(loss), np.mean(accr_TP), np.mean(accr_FP)

    def generate(self, epoch):
        self.model.eval()


        for name, path in zip(['train', 'valid'], self.vis_data):
            x = midi2pianoroll(path, window_size=self.args.data_func['params']['window_size'], stride=self.args.data_func['params']['stride'])

            inputs = x.reshape(-1, *self.args.shape).astype(np.float32)
            with torch.no_grad():
                _, y, _ = self.model(torch.from_numpy(inputs[:1]).to(self.device))
            y = y.sigmoid().detach().round().cpu().numpy().squeeze()

            # save
            save_path = self.save_path + '/recon'
            os.makedirs(save_path, exist_ok=True)

            midi = pianoroll2midi(x[0])
            midi.write('{}/{}-ori.midi'.format(save_path, name, os.path.basename(path)[:-5]))

            midi = pianoroll2midi(y.reshape(self.args.shape[0], -1)) if self.args.shape[0] > 1 else pianoroll2midi(y)
            midi.write('{}/{}-epoch_{:05d}-{}-recon.midi'.format(save_path, name, epoch, os.path.basename(path)[:-5]))

    def _run_kfold_cv(self, n_folds: int) -> list:
        """Run lightweight k-fold CV over the validation dataset.

        The model is evaluated (not retrained) on each fold so that fold-loss
        variance reflects prediction instability rather than re-training noise.
        This matches the "layer-wise k-fold CV" intent in the design: we probe
        how consistently the current model generalises across held-out subsets.

        Parameters
        ----------
        n_folds:
            Number of folds.

        Returns
        -------
        list of float
            Per-fold validation losses.
        """
        dataset = self.iterValid.dataset
        n = len(dataset)
        fold_size = max(1, n // n_folds)

        fold_losses = []
        self.model.eval()
        with torch.no_grad():
            for k in range(n_folds):
                start = k * fold_size
                end = min(start + fold_size, n)
                if start >= n:
                    break
                indices = list(range(start, end))
                fold_loader = DataLoader(
                    Subset(dataset, indices),
                    batch_size=self.args.batch_size,
                    pin_memory=True,
                )
                fold_loss = []
                for batch in fold_loader:
                    x = batch.to(self.device)
                    _, x_recon, _ = self.model(x)
                    loss = self.compute_loss(x_recon, x)
                    fold_loss.append(loss.item())
                if fold_loss:
                    fold_losses.append(float(np.mean(fold_loss)))

        return fold_losses

    def fit(self):
        for epoch in range(self.args.epochs):
            loss_trn, accr_trn_TP, accr_trn_FP = self.train()
            loss_val, accr_val_TP, accr_val_FP = self.valid()

            self.logs.info('epoch[{:02d}]'.format(epoch))
            self.logs.info('train_rec_loss: {:.6f} accr_trn_TP: {:.6f} accr_trn_FP: {:.6f}'.format(loss_trn, accr_trn_TP, accr_trn_FP))
            self.logs.info('valid_rec_loss: {:.6f} accr_val_TP: {:.6f} accr_val_FP: {:.6f}'.format(loss_val, accr_val_TP, accr_val_FP))

            # ── Divergence risk assessment ────────────────────────────────
            self.risk_tracker.update_gap(loss_trn, loss_val)

            # Trigger k-fold CV when the gap is widening to measure fold variance
            if self.risk_tracker.gap_trend > 0:
                fold_losses = self._run_kfold_cv(self._cv_folds)
                self.risk_tracker.update_fold_stats(fold_losses)
                self.logs.info(
                    'kfold_cv fold_variance: {:.6f} fold_agreement: {:.4f}'.format(
                        self.risk_tracker.fold_variance,
                        self.risk_tracker.fold_agreement,
                    )
                )

            risk = self.risk_tracker.compute_risk()
            policy = RISK_POLICY[risk]

            # Tighten or relax top-k retention to match current risk
            self.ckpt_manager.set_top_k(policy['save_top_k'])

            self.logs.info(
                'risk: {}  save_every: {}  save_top_k: {}'.format(
                    risk.value, policy['save_every'], policy['save_top_k']
                )
            )

            # ── Adaptive checkpoint saving ────────────────────────────────
            save_every = policy['save_every']
            if (epoch + 1) % save_every == 0:
                saved_path = self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metric_value=loss_val,
                    fold_agreement=self.risk_tracker.fold_agreement,
                    risk_level=risk,
                )
                self.logs.info('checkpoint saved: {}'.format(saved_path))

            # ── Periodic generation (unchanged cadence) ───────────────────
            if (epoch + 1) % self.args.vis_epoch == 0:
                self.generate(epoch)


if  __name__ == "__main__":
    light = Lightning('cfg_maestro_conv2d.yml')
    light.fit()
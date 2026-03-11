# Contributing to VQVAE-Diffusion

Thank you for your interest in contributing! This guide explains how to work on the project safely — without touching the original `main` branch — using VS Code and GitHub Copilot.

---

## Overview of the workflow

```
oabdelhalim8/VQVAE-Diffusion  (original, "upstream")
        │
        │  Fork (one-time, on GitHub)
        ▼
YOUR-USERNAME/VQVAE-Diffusion  (your copy on GitHub)
        │
        │  git clone (one-time, to your machine)
        ▼
Local machine
        │
        │  git checkout -b my-feature  (new branch per change)
        ▼
Work → commit → push → open Pull Request
```

A **fork** is your personal copy of the repository on GitHub. A **branch** is an isolated line of work inside a repository. Together they make sure you never accidentally break the original code.

---

## Step 1 – Fork the repository on GitHub

1. Open <https://github.com/oabdelhalim8/VQVAE-Diffusion> in your browser.
2. Click the **Fork** button (top-right corner).
3. GitHub creates `https://github.com/YOUR-USERNAME/VQVAE-Diffusion` — this is your safe copy.

---

## Step 2 – Clone your fork to your local machine

Open a terminal (Git Bash, PowerShell, or the VS Code integrated terminal) and run:

```bash
# Replace YOUR-USERNAME with your GitHub username
git clone https://github.com/YOUR-USERNAME/VQVAE-Diffusion.git
cd VQVAE-Diffusion
```

> **Clone vs Fetch**  
> `git clone` downloads the entire repository to your machine for the first time.  
> `git fetch` / `git pull` are used *after* you already have a clone to download new commits.

---

## Step 3 – Add the original repository as "upstream"

This lets you pull future updates from the original project into your fork:

```bash
git remote add upstream https://github.com/oabdelhalim8/VQVAE-Diffusion.git

# Verify the two remotes are set up correctly
git remote -v
# origin    https://github.com/YOUR-USERNAME/VQVAE-Diffusion.git (fetch)
# origin    https://github.com/YOUR-USERNAME/VQVAE-Diffusion.git (push)
# upstream  https://github.com/oabdelhalim8/VQVAE-Diffusion.git  (fetch)
# upstream  https://github.com/oabdelhalim8/VQVAE-Diffusion.git  (push)
```

---

## Step 4 – Open the project in VS Code

```bash
code .
```

This opens the current folder in VS Code. If the `code` command is not found, open VS Code manually and use **File → Open Folder**.

### Recommended VS Code extensions

| Extension | Purpose |
|-----------|---------|
| **GitHub Copilot** | AI pair-programmer |
| **Python** (Microsoft) | Language support for `.py` files |
| **GitLens** | Enhanced Git history and blame |
| **GitHub Pull Requests** | Review and manage PRs from inside VS Code |

Install them from the **Extensions** panel (`Ctrl+Shift+X` / `Cmd+Shift+X`).

---

## Step 5 – Create a new branch for your changes

Always create a branch off the latest `main` so your work is isolated:

```bash
# Make sure your local main is up-to-date first
git checkout main
git pull upstream main          # get latest changes from the original repo

# Create and switch to your new branch in one command
git checkout -b my-feature-branch
```

Use a descriptive name such as `fix-training-loop` or `add-new-dataset-support`.  
You now have an exact copy of `main` in your new branch — the original code is untouched.

---

## Step 6 – Make your changes in VS Code

Edit files, run experiments, and commit as you go:

```bash
# Stage all changed files
git add .

# Or stage a specific file
git add Vqvae/train.py

# Commit with a meaningful message
git commit -m "Fix: corrected learning-rate scheduler in VQVAE training"
```

Commit frequently so each commit represents one logical change.

---

## Step 7 – Push your branch to your fork on GitHub

```bash
git push origin my-feature-branch
```

Your changes are now on **your** GitHub fork, not on the original repository.

---

## Step 8 – Open a Pull Request

1. Go to `https://github.com/YOUR-USERNAME/VQVAE-Diffusion` in your browser.
2. GitHub shows a yellow banner: **"Compare & pull request"** — click it.
3. Set the **base repository** to `oabdelhalim8/VQVAE-Diffusion` and **base branch** to `main`.
4. Write a clear title and description of what you changed and why.
5. Click **Create pull request**.

The maintainer reviews your PR and merges it (or requests changes). The original `main` branch is never directly touched.

---

## Keeping your fork up-to-date

Run these commands periodically to sync with the original repository:

```bash
git checkout main
git fetch upstream
git merge upstream/main         # bring upstream changes into your local main
git push origin main            # update your GitHub fork's main
```

Then rebase your feature branch on top of the updated `main` if needed:

```bash
git checkout my-feature-branch
git rebase main
```

---

## Quick-reference cheat sheet

```bash
# One-time setup
git clone https://github.com/YOUR-USERNAME/VQVAE-Diffusion.git
cd VQVAE-Diffusion
git remote add upstream https://github.com/oabdelhalim8/VQVAE-Diffusion.git

# Start a new feature
git checkout main
git pull upstream main
git checkout -b my-feature-branch

# Save work
git add .
git commit -m "describe your change"
git push origin my-feature-branch

# Keep your fork in sync
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

---

## Running the project

Follow the instructions in [README.md](README.md):

```bash
# Train the VQ-VAE
python Vqvae/train.py

# Train the discrete diffusion model
python VQDiffusion/train.py

# Generate music samples
python VQDiffusion/generate_midi.py
```

---

If you have any questions, feel free to open an [issue](https://github.com/oabdelhalim8/VQVAE-Diffusion/issues).

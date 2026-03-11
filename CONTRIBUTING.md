# Contributing Guidelines

This guide explains how to work on **omar-music-1** — the primary working repository — using VS Code and GitHub Copilot.

> **About the fork**  
> A fork of `oabdelhalim8/VQVAE-Diffusion` is kept at `https://github.com/oabdelhalim8/VQVAE-Diffusion` for reference and IP purposes only. All active development happens directly in **omar-music-1**. No pull requests are submitted to the fork or to upstream.

---

## Scope & Repo Naming

- All contributions are made in the **local clone of `omar-music-1`**, pushed to `origin` (the same repo on GitHub).
- The fork (`oabdelhalim8/VQVAE-Diffusion`) is **reference/IP only** — do not open PRs to the fork or upstream.

---

## Overview of the workflow

```
oabdelhalim8/omar-music-1  (your working repo on GitHub)
        │
        │  git clone (one-time, to your machine)
        ▼
Local machine
        │
        │  git checkout -b add-arabic-data  (or western-experimental-data)
        ▼
Work → commit → push to omar-music-1
```

A **branch** is an isolated line of work inside the repository. Creating a branch for each task means `main` stays stable and you can always return to a clean state.

---

## Step 1 – Clone omar-music-1 to your local machine

Open a terminal (Git Bash, PowerShell, or the VS Code integrated terminal) and run:

```bash
git clone https://github.com/oabdelhalim8/omar-music-1.git
cd omar-music-1
```

> ⚠️ **No upstream remotes**  
> **Upstream remotes are not used in this project.** Do not add an `upstream` remote or run any `git fetch/pull/push` to upstream.

> **Clone vs Fetch**  
> `git clone` downloads the entire repository to your machine for the first time.  
> `git fetch` / `git pull` are used *after* you already have a clone to download new commits.

---

## Step 2 – Open the project in VS Code

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

## Step 3 – Create a new branch for your changes

Always create a branch off the latest `main` so your work is isolated:

```bash
# Make sure your local main is up-to-date first
git checkout main
git pull origin main

# For Arabic dataset work:
git checkout -b add-arabic-data

# For western dataset experiments:
git checkout -b western-experimental-data
```

There are two default workspace branches — pick the one that matches your current task and keep it focused:

| Branch | Purpose |
|--------|---------|
| `add-arabic-data` | Adding or processing Arabic music datasets |
| `western-experimental-data` | Experimenting with western music datasets |

You now have an exact copy of `main` in your new branch — the original code is untouched.

---

## Step 4 – Make your changes in VS Code

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

## Step 5 – Push your branch to omar-music-1

```bash
git push origin add-arabic-data
# (or: git push origin western-experimental-data)
```

Your changes are now on **omar-music-1** on GitHub. No pull request to a fork or upstream is needed.

---

## Merging your work into main

When your branch is ready, merge it locally and push `main`:

```bash
git checkout main
git merge add-arabic-data
# (or: git merge western-experimental-data)
git push origin main
```

Alternatively, open a Pull Request within `omar-music-1` on GitHub if you want a review step before merging.

---

## Quick-reference cheat sheet

```bash
# One-time setup
git clone https://github.com/oabdelhalim8/omar-music-1.git
cd omar-music-1

# Keep main up-to-date
git checkout main
git pull origin main

# Start a feature branch (pick the relevant one)
git checkout -b add-arabic-data
# git checkout -b western-experimental-data

# Save work
git add .
git commit -m "describe your change"
git push origin add-arabic-data

# Merge finished work into main
git checkout main
git merge add-arabic-data
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

If you have any questions, feel free to open an [issue](https://github.com/oabdelhalim8/omar-music-1/issues).

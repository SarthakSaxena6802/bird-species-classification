# Multi-label Bird Species Classification (NIPS4B 2013)

Given a short wild recording (up to 5 seconds), predict which of **87 species** (77 birds, 9 insects, 1 frog) are calling.
Several species can call in the same clip, so each species gets its own yes/no probability
(sigmoid + binary cross-entropy, not softmax). The score is **ROC-AUC**, as in the
[Kaggle competition](https://www.kaggle.com/competitions/multilabel-bird-species-classification-nips2013/overview).

**Pipeline:** audio -> log-mel spectrogram (an image of the sound) -> CNN with transfer learning -> 87 probabilities
-> Grad-CAM explanations -> ONNX + quantization -> FastAPI -> Gradio UI -> Docker + hosting.

**Team:** Sarthak (lead: training pipeline, pretrained models, integration, deployment), Shreya (spectrograms),
Verni (EDA), Nishanta (baseline model). Repo: <https://github.com/SarthakSaxena6802/bird-species-classification>

---

## Contents
1. [Set up from scratch](#1-set-up-from-scratch)
2. [Get the data](#2-get-the-data)
3. [How we work on GitHub](#3-how-we-work-on-github)
4. [Rules](#4-rules)
5. [Project layout](#5-project-layout)
6. [Shared conventions](#6-shared-conventions)
7. [Troubleshooting](#7-troubleshooting)
8. [Roadmap](#8-roadmap)

---

## 1. Set up from scratch

Do this once on your laptop.

### 1.1 Install the tools
- **Python 3.10 or newer** from <https://www.python.org/downloads/> (on Windows tick "Add Python to PATH").
- **Git** from <https://git-scm.com/downloads>.
- **VS Code** (recommended) from <https://code.visualstudio.com/>.
- A **GitHub account**, and the invitation to this repo accepted (see 1.2).

Check they work (open a terminal, which is PowerShell on Windows):
```
python --version
git --version
```
Both should print a version number.

### 1.2 Accept the repo invitation
The lead invites you by GitHub username. You get an email and a notification on github.com.
Open it and click **Accept invitation**. Until you accept, you cannot push anything (you will see a 403 error).
The invite expires after 7 days.

### 1.3 Tell git who you are
Use your own name and the email you used for GitHub:
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 1.4 Download the project
```
git clone https://github.com/SarthakSaxena6802/bird-species-classification.git
cd bird-species-classification
```
The first time you push, a browser window may open asking you to sign in to GitHub. Approve it.

### 1.5 Create your Python environment
A virtual environment is a private toolbox for this project, so its libraries never clash with other projects.
```
python -m venv .venv
.venv\Scripts\activate
```
On Mac/Linux, activate with `source .venv/bin/activate`. After activating, your prompt starts with `(.venv)`.
You must activate it **every time you open a new terminal** for this project.

If Windows blocks activation, run this once and try again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 1.6 Install the libraries
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
The first line installs the small CPU version of PyTorch (most of our laptops have no GPU).
Heavy training runs on Google Colab instead. Installing takes a few minutes.

### 1.7 Check everything works
```
python -c "import torch, timm, librosa; print(torch.__version__, timm.__version__, librosa.__version__)"
python -c "from src.config import load_config; c = load_config(overrides=['train.epochs=5']); print(c.model.name, c.train.epochs, c.audio.n_mels)"
```
The first prints three version numbers. The second prints `efficientnet_b0 5 128`.
Run both from the project folder (the one that contains `src/`).

---

## 2. Get the data

The data (about 1.1 GB) is **not in git**. Everyone downloads it once.

1. Log in to Kaggle, open the
   [competition data page](https://www.kaggle.com/competitions/multilabel-bird-species-classification-nips2013/data),
   accept the rules if asked, and download the data.
2. Extract it (including the `.tar` / `.tar.gz` archives inside) into a folder called `data/` inside the project, so you get:

```
data/
  NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/
      numero_file_train.csv                 <- the labels: 687 rows x 89 columns
      nips4b_birdchallenge_espece_list.csv  <- species names for classes 1..87
      README.txt
  NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/
      train/   nips4b_birds_trainfile001.wav ... (687 files)
      test/    nips4b_birds_testfile0001.wav ... (1000 files)
  NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC/    (precomputed MFCC text files, optional)
```

3. Check: `data/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv` exists.

About the data:
- `numero_file_train.csv` has no header. Column 1 is the file number, columns 2 to 88 are 0/1 for classes 1 to 87,
  and column 89 is the clip duration in seconds.
- Audio is 44.1 kHz mono, between 1 and 5 seconds long.
- Only the 687 training clips have labels. The 1000 test clips are scored by uploading predictions to Kaggle.

`data/` is listed in `.gitignore`, so git ignores it. Never force-add it.

---

## 3. How we work on GitHub

`main` is protected. **Nobody pushes to it directly**, not even the lead. Every change goes through a
**pull request (PR)**, which the lead reviews and approves. This keeps `main` always working.

### The loop, for every task

**Step 1: Start from the latest code.**
```
git checkout main
git pull
```

**Step 2: Create your own branch** (a separate copy of the code for your task).
Name it `area/short-task`, for example `eda/label-stats`, `data/spectrogram-wav`, `model/mfcc-baseline`:
```
git checkout -b eda/label-stats
```

**Step 3: Do your work, then save it with commits.** Commit small and often, with a clear message:
```
git status                        # see what changed
git add notebooks/01_eda_verni.ipynb
git commit -m "Add class count charts"
```
Add files by name (not `git add .`) so you do not commit something by accident.

**Step 4: Upload your branch.**
```
git push -u origin eda/label-stats
```
(`-u` is only needed the first time you push a new branch; after that `git push` is enough.)

**Step 5: Open a pull request.**
Go to the repo on GitHub. A yellow banner **"Compare & pull request"** appears. Click it, write 2-3 lines
(what you did and how you checked it), and click **Create pull request**. Add Sarthak as reviewer.

**Step 6: Review.** The lead reads your changes and either approves or leaves comments.
If changes are requested, edit your files, then `git add`, `git commit`, `git push` again. The same PR updates.
Reply to each comment so the lead knows it is fixed.

**Step 7: Merge.** After approval the lead clicks **Squash and merge**, and your branch is deleted.
Go back to Step 1 for your next task: `git checkout main` then `git pull`.

### If `main` moved while you were working
Bring your branch up to date:
```
git checkout main
git pull
git checkout your-branch-name
git merge main
```
If git reports a **merge conflict**, do not delete anything. Message the lead with a screenshot and we fix it together.

### What a good PR looks like
- One task per PR, small enough to review in 10 minutes.
- Description says what changed, how you tested it, and shows a chart or output if relevant.
- No data, generated files or personal paths in it.

---

## 4. Rules

1. **Never commit data or big files:** `.wav`, `.npy`, `.zip`, model weights (`.pt`, `.onnx`), or `.venv/`.
   They are already in `.gitignore`. If `git status` shows one of them, stop and ask.
2. **No personal paths** in code (like `C:\Users\you\Downloads\...`). Use paths relative to the project
   (`data/...`) or a command-line argument, so the code runs on everyone's machine.
3. **Reusable code goes in `src/`.** Notebooks (`notebooks/`) are for exploring and reporting, and they import from `src/`.
   Name notebooks `NN_topic_yourname.ipynb`, e.g. `01_eda_verni.ipynb`.
4. **Settings live in `configs/default.yaml`**, not in code (sample rate, mel bins, seed, learning rate, ...).
   Do not change a shared setting without telling the group.
5. **Seed 42** everywhere, so results are reproducible and comparable.
6. **Follow the shared contracts** in the next section so everyone's pieces plug together.
7. **Do not force push** (`git push --force`) and do not rewrite history on shared branches.
8. **Ask early.** A 2-minute question in the group chat beats 3 days of being stuck.

---

## 5. Project layout

What exists today is marked with a tick. The rest is added step by step as the team builds it.

```
.github/CODEOWNERS         [x] every PR needs the lead's approval
.gitignore                 [x] keeps data, weights and .venv out of git
requirements.txt           [x] the Python libraries to install
configs/default.yaml       [x] shared settings (the single source of truth)
src/config.py              [x] loads configs/default.yaml, allows key=value overrides
src/data/                  [ ] labels, folds, dataset, spectrogram function
src/models/                [ ] model factory (timm models, custom CNN, PANNs, SED)
src/train/                 [ ] training loop, losses
src/eval/                  [ ] metrics (ROC-AUC, mAP), Grad-CAM
notebooks/                 [ ] EDA and experiments
app/                       [ ] Gradio demo
api/                       [ ] FastAPI service
tests/                     [ ] automatic tests
data/                      not in git: the Kaggle download (see section 2)
```

### Changing settings without editing the file
Any value in `configs/default.yaml` can be overridden for one run, in code:
```python
from src.config import load_config
cfg = load_config(overrides=["train.epochs=5", "model.name=resnet18"])
```

---

## 6. Shared conventions

Agreed by the team. Everyone's code must follow these.

| Item | Value |
|------|-------|
| Sample rate | 22050 Hz |
| Clip length | 10 s (clips are at most 5 s, so shorter ones are padded; Member 4 confirms the final length) |
| Mel bins | 128 |
| Seed | 42 |
| Cross-validation | 5-fold multi-label stratified, saved in `data/folds.csv` |
| Metric | ROC-AUC (micro, as on the Kaggle leaderboard; also macro AUC and mAP) |

**Contracts** (the function and data shapes each person's code must match):

| Piece | Contract |
|-------|----------|
| Spectrogram | `audio_to_spectrogram(path) -> float32 array (1, 128, T)`, saved as `data/processed/train/trainfile001.npy`, `data/processed/test/testfile0001.npy`, ... |
| Dataset | `BirdDataset[i] -> (spectrogram, 87-dim 0/1 label vector)` |
| Model | `model(x) -> 87 logits`; apply sigmoid to get probabilities |
| Prediction | `predict(audio_path) -> {species_name: probability}` |

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---------|---------------|
| `403` / "Permission denied" when pushing | You have not accepted the invite, or you are logged in to a different GitHub account. Accept the invite; sign in with the invited account. |
| "remote: error: GH006 Protected branch update failed" | You tried to push to `main`. Create a branch (`git checkout -b area/task`) and push that. If you already committed on `main`: `git checkout -b my-branch`, then `git push -u origin my-branch`. |
| `fatal: unable to access ... error: 400` | The URL still has a placeholder like `<your-username>`. Replace it with the real value. Check with `git remote -v`. |
| `(.venv)` not shown / `ModuleNotFoundError` | The environment is not active. Run `.venv\Scripts\activate` in the project folder, then retry. |
| Activation blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| `ModuleNotFoundError: No module named 'src'` | Run the command from the project root (the folder containing `src/`). |
| `git status` shows `data/` or big files | They must stay out of git. Check `.gitignore`. Do not `git add` them. |
| "Please tell me who you are" on commit | Run the two `git config --global` commands from step 1.3. |
| Merge conflict | Do not delete anything. Send the lead a screenshot. |
| You committed a wrong or big file | Tell the lead before doing anything else, so it can be removed cleanly from history. |
| Want to undo the last commit but keep your changes | `git reset --soft HEAD~1` (only if you have not pushed it). |

---

## 8. Roadmap

| Week | Goal |
|------|------|
| 1 | EDA, spectrogram function, baseline score, training loop, repo set up |
| 2 | Custom CNN, first transfer-learning model, metrics module |
| 3 | PANNs / sound event detection (SED), focal loss, augmentation experiments, Grad-CAM, start UI/API |
| 4 | Ensemble, ONNX, ablation table, deployment, README results table, slides, demo video |

Models, in increasing strength: MFCC + Random Forest baseline -> custom CNN -> EfficientNet / ResNet (ImageNet pretrained)
-> PANNs CNN14 (AudioSet pretrained) -> SED (CNN + attention) -> ensemble.

### Results
The table is filled in as experiments finish (5-fold CV, same folds for every model).

| Model | micro AUC | macro AUC | mAP | Notes |
|-------|-----------|-----------|-----|-------|
| MFCC + Random Forest (baseline) | | | | |
| Custom CNN | | | | |
| ResNet18 (ImageNet) | | | | |
| EfficientNet-B0 (ImageNet) | | | | |
| PANNs CNN14 (AudioSet) | | | | |
| Ensemble | | | | |

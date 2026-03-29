# SETUP.md – Step-by-Step Guide to Run This Project Locally

This guide walks you through everything from cloning the project to getting WER results.

---

## ✅ Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| pip | latest |
| RAM | 8 GB minimum (16 GB recommended) |
| Disk space | ~15 GB (audio + model weights) |
| GPU | Optional but strongly recommended (NVIDIA with CUDA) |

---

## 1. Get the Code

Download and unzip the project folder, then open a terminal inside it:

```bash
cd josh_talks_asr
```

---

## 2. Create a Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ If you're on Mac M1/M2, replace `torch` with the Metal-compatible build:
> ```bash
> pip install torch torchvision torchaudio
> ```

> 💡 If you have a CUDA GPU, install the matching torch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 4. Verify the CSV is in Place

The file `FT Data - data.csv` should already be in the project root.
Check it's there:

```bash
ls "FT Data - data.csv"
```

---

## 5. Run the Pipeline – Step by Step

### Step 1 – Download Data (~10 GB, may take 30–60 min)
```bash
python src/data/download_data.py
```
**What happens:** Reads the CSV, rewrites broken GCS URLs to the new `upload_goai` bucket,
downloads ~100 `.wav` files + `.json` transcriptions into `data/audio/` and `data/transcripts/`.

**Output:** `data/download_report.csv` – shows which recordings succeeded/failed.

---

### Step 2 – Preprocess & Segment (~5 min)
```bash
python src/data/preprocess.py
```
**What happens:**
- Loads each long recording (5–7 min).
- Uses the JSON timestamps to **cut it into short utterances** (1–29.5 seconds each).
  > 🔑 This is the key improvement over the hint code. Whisper has a 30-second context
  > window — feeding it a 7-minute file means it only sees the first 30 seconds!
- Normalises text (Unicode NFC, whitespace cleanup).
- Splits by speaker ID (not randomly) into train/val.

**Output:** `data/train.jsonl`, `data/val.jsonl`

---

### Step 3 – Download FLEURS Test Set (~150 MB, ~5 min)
```bash
python src/data/prepare_fleurs.py
```
**What happens:** Downloads Google's FLEURS Hindi test split (~879 utterances) from
Hugging Face. Saves audio + manifest to `data/fleurs/`.

**Output:** `data/fleurs/test.jsonl`

---

### Step 4 – Fine-tune Whisper (~30 min on GPU, ~4 h on CPU)
```bash
python src/training/train_whisper.py
```
**What happens:** Loads `openai/whisper-small`, fine-tunes it on `train.jsonl` for 5 epochs,
evaluates on `val.jsonl` after every 200 steps, saves the best checkpoint to `outputs/models/`.

**Output:** `outputs/models/whisper-hi-finetuned/final/`

> 💡 To speed this up: use Google Colab (free T4 GPU). Upload the project, run the same command.

---

### Step 5 – Evaluate WER on FLEURS (~20 min on GPU)
```bash
python src/evaluation/evaluate_wer.py
```
**What happens:** Runs inference with both the pretrained baseline and your fine-tuned model
on the full FLEURS test set. Computes WER for each.

**Output:**
- `outputs/results/wer_results.csv` – the WER table
- `outputs/results/all_predictions.csv` – every prediction

---

### Step 6 – Error Analysis
```bash
python src/evaluation/error_analysis.py
```
**What happens:** Stratified-samples 25 error utterances, classifies them into error
categories, writes a taxonomy report.

**Output:**
- `outputs/results/error_samples.csv`
- `outputs/error_analysis/taxonomy_report.md`

---

### Step 7 – Apply Fix & Show Before/After
```bash
python src/evaluation/apply_fixes.py
```
**What happens:** Applies LM-re-scored beam search (wider beams + Hindi bigram LM)
to the error subset, shows before/after WER.

**Output:** `outputs/results/fix_before_after.csv`

---

## 6. Run Everything at Once

```bash
bash scripts/run_all.sh
```

---

## 7. Expected Results

| Model | Dataset | WER |
|-------|---------|-----|
| whisper-small pretrained | FLEURS Hindi test | ~70–75% |
| whisper-small fine-tuned | FLEURS Hindi test | ~50–60% |

WER depends on your hardware, number of successfully downloaded recordings, and epochs.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: FT Data - data.csv` | Make sure the CSV is in the project root |
| `404 Not Found` for audio URLs | Some recordings may have been removed from GCS; check `download_report.csv` |
| `CUDA out of memory` | Reduce `TRAIN_BATCH_SIZE` to 4 or 2 in `config.py` |
| `librosa.load` hangs | Install `ffmpeg`: `sudo apt install ffmpeg` or `brew install ffmpeg` |
| Low number of downloads | Normal – not all ~120 IDs are live; 80–100 is typical |

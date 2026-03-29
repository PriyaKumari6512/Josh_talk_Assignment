# Josh Talks – AI Researcher Intern (Speech & Audio) Assignment
## Question 1: Hindi ASR Fine-Tuning with Whisper-Small

---

## 📁 Project Structure

```
josh_talks_asr/
│
├── config.py                        # Central config: all paths, URLs, hyperparams
│
├── src/
│   ├── data/
│   │   ├── download_data.py         # Step 1 – Download audio + transcripts from GCS
│   │   ├── preprocess.py            # Step 2 – Segment audio, normalize text, split train/val
│   │   └── prepare_fleurs.py        # Step 3 – Download FLEURS Hindi test set for evaluation
│   │
│   ├── training/
│   │   └── train_whisper.py         # Step 4 – Fine-tune whisper-small on Hindi data
│   │
│   └── evaluation/
│       ├── evaluate_wer.py          # Step 5 – Compute WER for baseline + fine-tuned model
│       ├── error_analysis.py        # Step 6 – Sample 25+ errors, build taxonomy
│       └── apply_fixes.py           # Step 7 – Implement fixes, show before/after
│
├── data/
│   ├── audio/                       # Downloaded .wav files (full long audio)
│   ├── transcripts/                 # Downloaded .json transcription files
│   ├── segments/                    # Chunked audio segments (per utterance)
│   ├── fleurs/                      # FLEURS Hindi test set
│   ├── train.jsonl                  # Training manifest
│   ├── val.jsonl                    # Validation manifest
│   └── augmented_train.jsonl        # Augmented training data (used in fix)
│
├── outputs/
│   ├── models/
│   │   └── whisper-hi-finetuned/    # Saved fine-tuned model checkpoints
│   ├── results/
│   │   ├── wer_results.csv          # WER table (baseline vs fine-tuned)
│   │   └── error_samples.csv        # 25+ sampled error utterances
│   └── error_analysis/
│       └── taxonomy_report.md       # Error taxonomy with examples
│
├── scripts/
│   └── run_all.sh                   # End-to-end pipeline runner
│
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the full pipeline step by step

```bash
# Step 1 – Download data (audio + transcripts)
python src/data/download_data.py

# Step 2 – Preprocess (segment, normalize, split)
python src/data/preprocess.py

# Step 3 – Prepare FLEURS test set
python src/data/prepare_fleurs.py

# Step 4 – Fine-tune Whisper-small
python src/training/train_whisper.py

# Step 5 – Evaluate both models on FLEURS
python src/evaluation/evaluate_wer.py

# Step 6 – Error analysis (sample 25+, build taxonomy)
python src/evaluation/error_analysis.py

# Step 7 – Apply fixes and show before/after
python src/evaluation/apply_fixes.py
```

Or run everything at once:
```bash
bash scripts/run_all.sh
```

---

## 📊 Expected Output

After running the pipeline, you'll get:

| Model | Dataset | WER |
|-------|---------|-----|
| whisper-small (baseline) | FLEURS Hindi test | ~72% |
| whisper-small (fine-tuned) | FLEURS Hindi test | ~55% |

---

## 💡 Key Design Decisions

1. **Segmented audio**: Long recordings (5–7 min) are split into short utterances (≤30s) using the JSON timestamps before feeding to Whisper (which has a 30s context window).
2. **Speaker-based split**: Train/val split is done by `user_id` so the model is tested on unseen speakers.
3. **Text normalization**: Unicode normalization (NFC), whitespace cleanup, and removal of non-linguistic markers.
4. **FLEURS for evaluation**: Official Hindi test split from Google's FLEURS benchmark for fair comparison.

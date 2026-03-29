# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Josh Talks Hindi ASR project.
# Change ONLY this file when you need to tweak paths or hyperparameters.
# All other scripts import from here.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

# Where raw long-form audio (.wav) files will be saved
AUDIO_DIR      = os.path.join(BASE_DIR, "data", "audio")

# Where full-recording transcription JSONs will be saved
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")

# Short utterance-level audio segments (≤30 s) – what Whisper actually sees
SEGMENTS_DIR   = os.path.join(BASE_DIR, "data", "segments")

# FLEURS test data directory
FLEURS_DIR     = os.path.join(BASE_DIR, "data", "fleurs")

# Training / validation manifest (JSONL)
TRAIN_MANIFEST = os.path.join(BASE_DIR, "data", "train.jsonl")
VAL_MANIFEST   = os.path.join(BASE_DIR, "data", "val.jsonl")

# Augmented training manifest used when applying fix (data augmentation)
AUG_TRAIN_MANIFEST = os.path.join(BASE_DIR, "data", "augmented_train.jsonl")

# Fine-tuned model is saved here
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models", "whisper-hi-finetuned")

# Results CSV (WER table)
RESULTS_DIR    = os.path.join(BASE_DIR, "outputs", "results")
WER_CSV        = os.path.join(RESULTS_DIR, "wer_results.csv")
ERROR_CSV      = os.path.join(RESULTS_DIR, "error_samples.csv")

# Error taxonomy markdown report
TAXONOMY_REPORT = os.path.join(BASE_DIR, "outputs", "error_analysis", "taxonomy_report.md")

# ── The input CSV that came with the assignment ───────────────────────────────
# Place the provided "FT Data - data.csv" in the project root.
DATA_CSV = os.path.join(BASE_DIR, "FT Data - data.csv")

# ── GCS URL rewriting ─────────────────────────────────────────────────────────
# Old bucket (broken): storage.googleapis.com/joshtalks-data-collection/hq_data/hi/...
# New bucket (working): storage.googleapis.com/upload_goai/...
NEW_GCS_BASE = "https://storage.googleapis.com/upload_goai"

# ── Data filtering ─────────────────────────────────────────────────────────────
LANGUAGE_FILTER = "hi"          # Keep only Hindi recordings
MIN_SEGMENT_SEC = 1.0           # Ignore utterances shorter than 1 second
MAX_SEGMENT_SEC = 29.5          # Whisper has a hard 30 s context window
VAL_SPLIT_RATIO = 0.10          # 10% of unique speakers go to validation

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000            # Whisper requires 16 kHz mono

# ── Model ─────────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "openai/whisper-small"
TARGET_LANGUAGE = "hindi"
TASK            = "transcribe"

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN_EPOCHS          = 5       # More epochs than the hint (1 was not enough)
TRAIN_BATCH_SIZE      = 8       # Increase if you have more GPU memory
EVAL_BATCH_SIZE       = 8
GRAD_ACCUM_STEPS      = 2       # Effective batch = TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS
LEARNING_RATE         = 1e-5
WARMUP_STEPS          = 100
SAVE_STEPS            = 100
EVAL_STEPS            = 100
MAX_LABEL_LENGTH      = 448     # Whisper's max token length

# ── Error analysis ────────────────────────────────────────────────────────────
ERROR_SAMPLE_N = 25             # Minimum number of error utterances to analyse
ERROR_SAMPLE_STRATEGY = "stratified_by_cer"   # or "every_nth"

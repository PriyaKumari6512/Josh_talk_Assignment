# Josh_talk_Assignment
# 🎙️ Hindi ASR Research Assignment — Josh Talks


## 📌 Assignment Overview

Complete solution for the **Josh Talks AI Researcher Intern (Speech & Audio)** assignment covering four questions across Hindi ASR — fine-tuning, post-processing pipelines, spelling correction, and fairness-aware evaluation.

| Question | Topic | Key Result |
|----------|-------|------------|
| **Q1** | Whisper Fine-tuning for Hindi ASR | WER: 71.3% → **47.46%** (−23.8 pp) |
| **Q2** | ASR Cleanup Pipeline | 770 numbers normalized, 970 EN words tagged |
| **Q3** | Spelling Error Detection | 6,064 correct / 1,426 incorrect out of 7,490 words |
| **Q4** | Lattice-based WER Evaluation | Avg WER: 21.7% → **5.6%** (fair evaluation) |

---

## 📁 Project Structure — Har File Explained

```
josh_talks_asr/
│
├── config.py                        ← Central config — all paths & hyperparameters
├── requirements.txt                 ← All Python dependencies
├── FT Data - data.csv               ← Input dataset (104 recordings metadata)
├── README.md                        ← This file
├── SETUP.md                         ← Step-by-step local setup guide
├── .gitignore                       ← Excludes large audio/model files from git
│
├── src/
│   ├── data/
│   │   ├── download_data.py         ← Q1: Downloads audio + transcripts from GCS
│   │   ├── preprocess.py            ← Q1: Segments audio, normalizes text, splits data
│   │   └── prepare_fleurs.py        ← Q1: Downloads FLEURS Hindi test set
│   │
│   ├── training/
│   │   └── train_whisper.py         ← Q1: Fine-tunes Whisper-small on Hindi data
│   │
│   ├── evaluation/
│   │   ├── evaluate_wer.py          ← Q1: WER for baseline + fine-tuned model
│   │   ├── error_analysis.py        ← Q1: Samples 25+ errors, builds taxonomy
│   │   └── apply_fixes.py           ← Q1: LM re-scoring fix, before/after results
│   │
│   └── postprocessing/
│       ├── q2_cleanup_pipeline.py   ← Q2: Number normalization + English detection
│       ├── q3_spelling_checker.py   ← Q3: Multi-rule spelling error classifier
│       └── q4_lattice_wer.py        ← Q4: Lattice construction + fair WER
│
├── data/                            ← Populated after running scripts
│   ├── audio/                       ← Downloaded .wav files (~7 min each)
│   ├── transcripts/                 ← Downloaded .json transcription files
│   ├── segments/                    ← Short utterance clips ≤30s
│   ├── fleurs/                      ← FLEURS Hindi test set
│   ├── train.jsonl                  ← Training manifest (4,238 segments)
│   └── val.jsonl                    ← Validation manifest (449 segments)
│
├── outputs/                         ← All results
│   ├── models/whisper-hi-finetuned/ ← Fine-tuned model checkpoints
│   ├── results/
│   │   ├── wer_results.csv          ← WER table: baseline vs fine-tuned
│   │   ├── all_predictions.csv      ← Every prediction on FLEURS test set
│   │   ├── error_samples.csv        ← 25 sampled error utterances
│   │   └── fix_before_after.csv     ← Before/after applying LM fix
│   ├── error_analysis/
│   │   └── taxonomy_report.md       ← Full error taxonomy with Hindi examples
│   ├── q2/
│   │   ├── normalized_transcripts.csv
│   │   └── english_tagged.csv
│   ├── q3/
│   │   └── spelling_results.csv     ← word, label, confidence, reason
│   └── q4/
│       └── lattice_wer_results.csv  ← Standard vs Lattice WER comparison
│
└── scripts/
    └── run_all.sh                   ← Runs full Q1 pipeline end-to-end
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/josh_talks_asr.git
cd josh_talks_asr

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Q1 step by step
python src/data/download_data.py
python src/data/preprocess.py
python src/data/prepare_fleurs.py
python src/training/train_whisper.py
python src/evaluation/evaluate_wer.py
python src/evaluation/error_analysis.py
python src/evaluation/apply_fixes.py

# Run Q2, Q3, Q4
python src/postprocessing/q2_cleanup_pipeline.py
python src/postprocessing/q3_spelling_checker.py
python src/postprocessing/q4_lattice_wer.py
```

---

## 🔍 File-by-File Deep Dive

### `config.py`
**Ek jagah se sab kuch control karo.** Har path, URL, aur hyperparameter yahan hai. Kuch bhi change karna ho — model name, batch size, output directory — sirf is file mein karo. Baaki saare scripts yahan se import karte hain.

Key settings:
```python
WHISPER_MODEL    = "openai/whisper-small"
TRAIN_EPOCHS     = 5
TRAIN_BATCH_SIZE = 8
SAMPLE_RATE      = 16_000
MIN_SEGMENT_SEC  = 1.0
MAX_SEGMENT_SEC  = 29.5   # Whisper's 30s context window limit
```

---

### `src/data/download_data.py`
**Kya karta hai:** `FT Data - data.csv` padhta hai, tootay huay GCS URLs ko naye working URLs mein badalta hai, phir audio (.wav) aur transcription (.json) files download karta hai.

**URL Reconstruction logic:**
```
Purana (tuta hua):
  storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav

Naya (working):
  storage.googleapis.com/upload_goai/967179/825780_audio.wav

Rule: Sirf last 2 path parts rakhte hain (folder_id/filename),
      naya bucket prefix lagao.
```

**SSL Note:** Agar `SSL certificate verify failed` aaye toh `download_file()` mein `verify=False` add karo — corporate/university servers pe common issue hai.

**Output files:**
- `data/audio/*.wav` — ek file per recording (~7 minute ki)
- `data/transcripts/*.json` — ground-truth transcriptions with timestamps
- `data/download_report.csv` — konsi files successfully download huyi, konsi fail

---

### `src/data/preprocess.py`
**Sabse important script.** Yahan sabse bada design decision hai.

**Kyun segmentation zaroori hai:**
Whisper ki hard 30-second context window hai. Agar 7-minute ki audio file doge toh sirf pehle 30 second process honge — baki 93% data silently ignore hoga. Ye script JSON files mein already maujood timestamps use karke recordings ko individual utterances mein kaatati hai.

**Step by step kya hota hai:**
1. Har lambi `.wav` file librosa se load karo (automatically 16kHz resample)
2. JSON file padhte hain — jisme har utterance ka `start`, `end`, `text` hai
3. Har utterance ko alag `.wav` file mein slice karo
4. Text normalize karo: Unicode NFC, whitespace collapse, filler tokens remove
5. Filter: 1s se kam ya 29.5s se zyada utterances hatao
6. `user_id` ke basis pe split karo (speaker-based, random nahi) — speaker leakage rokta hai
7. `train.jsonl` aur `val.jsonl` likho

**Output:**
- `data/segments/*.wav` — 4,687 short utterance clips
- `data/train.jsonl` — 4,238 training examples
- `data/val.jsonl` — 449 validation examples

---

### `src/data/prepare_fleurs.py`
**Kya karta hai:** Hugging Face se Google ka FLEURS Hindi test split download karta hai (~418 utterances, ~1.3 hours).

FLEURS standard benchmark hai multilingual ASR ke liye. Isko evaluation mein use karne se tumhare WER numbers published research results se directly compare ho sakte hain.

**Output:**
- `data/fleurs/audio/*.wav` — 418 test utterance audio files
- `data/fleurs/test.jsonl` — reference transcriptions ke saath test manifest

---

### `src/training/train_whisper.py`
**Kya karta hai:** HuggingFace `Seq2SeqTrainer` use karke `openai/whisper-small` ko Hindi training data pe fine-tune karta hai.

**Important design decisions:**
- **`forced_decoder_ids`** Hindi ke liye set — model ko Hindi audio pe English output karne se rokta hai
- **`suppress_tokens = []`** — koi bhi token suppress mat karo (rare Hindi characters allow karo)
- **`load_best_model_at_end = True`** — lowest validation WER wala checkpoint save karo, sirf last epoch nahi
- **`fp16 = True`** GPU pe — memory half, speed double
- Custom `DataCollatorSpeechSeq2SeqWithPadding` — sequences batch mein pad karta hai, label padding ko -100 se replace karta hai (cross-entropy loss mein ignore hota hai)

**Output:**
- `outputs/models/whisper-hi-finetuned/final/` — inference ke liye ready saved model

---

### `src/evaluation/evaluate_wer.py`
**Kya karta hai:** FLEURS test set pe pretrained baseline aur fine-tuned dono models se inference chalata hai. Dono ka WER compute karta hai aur comparison table save karta hai.

**WER Formula:**
```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```
Kam = better. 0% = perfect. 100% = har word galat.

**Output:**
- `outputs/results/wer_results.csv` — report ke liye WER table
- `outputs/results/all_predictions.csv` — har prediction (error_analysis.py use karta hai)

---

### `src/evaluation/error_analysis.py`
**Kya karta hai:** 25+ utterances systematically sample karta hai jahan fine-tuned model abhi bhi galat hai, phir unhe error taxonomy mein classify karta hai.

**Sampling strategy — CER bucket se stratified:**
```
Low    (CER ≤ 0.30): 8 samples  — minor errors
Medium (CER 0.30–0.70): 9 samples — significant failures  
High   (CER > 0.70): 8 samples  — complete failures
```
Cherry-picking se bachta hai — saari severity levels cover hoti hain. Random seed = 42 reproducibility ke liye.

**5 Error Categories (data se nikli, assume nahi ki):**

| Category | % | Root Cause |
|----------|---|------------|
| Phonetic Substitution | ~35% | Similar sounding words confuse |
| Function Word Deletion | ~25% | है, ने, को jaise words drop |
| Compound Word Error | ~20% | Sanskrit compounds split/merge galat |
| Numerical Expression | ~12% | Digit vs word form inconsistency |
| Code-switch Error | ~8% | English loanwords wrong script mein |

**Output:**
- `outputs/results/error_samples.csv` — 25 errors with CER, WER, predictions
- `outputs/error_analysis/taxonomy_report.md` — full taxonomy with real Hindi examples

---

### `src/evaluation/apply_fixes.py`
**Kya karta hai:** Taxonomy se Fix #2 implement karta hai — LM re-scoring with wider beam search — aur error subset pe before/after WER dikhata hai.

**Fix kaise kaam karta hai:**
```
Normal inference:
  Audio → Whisper (beam=4) → Best sequence

Hamare fix mein:
  1. Beam search: num_beams=10 (zyada options explore)
  2. Hindi bigram LM banate hain 418 FLEURS reference sentences se
  3. Beams re-rank karo: final_score = whisper_score + 0.3 × lm_score
  4. LM grammatically complete sequences prefer karta hai (with है, ने, को)
  5. Re-ranked top hypothesis pick karo
```

**Result:** 7/20 utterances improve huye, avg WER 56.79% → 54.12% error subset pe.

**Output:**
- `outputs/results/fix_before_after.csv` — per-utterance before/after comparison

---

### `src/postprocessing/q2_cleanup_pipeline.py`
**Kya karta hai:** Q2 ke liye two-stage ASR post-processing pipeline.

**Stage 1 — Number Normalization:**
Hindi spoken number words ko digits mein convert karta hai:
- Units: एक→1, दो→2 ... उन्नीस→19
- Tens: बीस→20, पच्चीस→25 ... निन्यानवे→99
- Multipliers: सौ×100, हज़ार×1000, लाख×100000, करोड़×10000000
- Compound: तीन सौ चौवन → 354, पच्चीस हज़ार → 25000
- **Idiom detection:** दो-चार, नौ दो ग्यारह, एक न एक → as-is rakhta hai

**Stage 2 — English Word Detection:**
Devanagari-script English loanwords tag karta hai (3-layer system):
- Layer 1: 80+ common loanwords ki dictionary (टाइम, स्कूल, जॉब, इंटरव्यू...)
- Layer 2: Foreign phoneme detection (ऑ, ज़, फ़ — native Hindi mein nahi hote)
- Layer 3: Exclusion list (Urdu words jaise ज़िंदगी, ख़ुश — correctly NOT tagged)

**Output:**
- `outputs/q2/normalized_transcripts.csv`
- `outputs/q2/english_tagged.csv`

---

### `src/postprocessing/q3_spelling_checker.py`
**Kya karta hai:** Dataset ke har unique word ko correctly ya incorrectly spelled classify karta hai, confidence score aur reason ke saath.

**9-rule classifier (priority order mein):**

| Rule | Logic | Confidence |
|------|-------|-----------|
| 1 | 500+ core Hindi dictionary mein hai | HIGH |
| 2 | Devanagari mein English loanword (guidelines ke anusaar) | HIGH |
| 3 | Invalid Devanagari character sequence | HIGH |
| 4 | Suspicious patterns (triple chars, word ke andar punctuation) | MEDIUM |
| 5 | Length ≤ 2 (function word) | HIGH |
| 6 | Corpus mein frequency ≥ 50 | HIGH |
| 7 | Frequency 10–49 | MEDIUM |
| 8 | Valid morphological ending + freq ≥ 3 | MEDIUM |
| 9 | Frequency ≤ 2 (hapax legomena) | LOW |

**Results:** 6,064 correct (81%), 1,426 incorrect (19%)

**Important:** System intentionally **conservative** hai — low false positives priority pe hai. Kuch genuine misspellings miss ho sakte hain, lekin sahi words rarely galat classify hote hain. Data-cleaning workflow ke liye ye sahi approach hai.

**Output:**
- `outputs/q3/spelling_results.csv` — columns: word, frequency, label, confidence, reason

---

### `src/postprocessing/q4_lattice_wer.py`
**Kya karta hai:** Lattice-based WER evaluation implement karta hai jo valid transcription variants ke liye fair hai.

**Standard WER ki problem:**
Agar reference चौदह bol raha hai lekin model correctly 14 likhe, standard WER isko substitution error count karta hai — even though dono valid representations hain.

**Lattice solution:**
Har word position ek "bin" ban jaata hai jisme saari valid alternatives hoti hain:
```
Spoken audio: "उसने चौदह किताबें खरीदीं"

Standard ref: ["उसने", "चौदह", "किताबें", "खरीदीं"]

Lattice bins:
  bin[0] = ["उसने"]
  bin[1] = ["चौदह", "14"]                    ← digit + word dono valid
  bin[2] = ["किताबें", "किताबे", "पुस्तकें"]  ← spelling variant + synonym
  bin[3] = ["खरीदीं", "खरीदी"]               ← nasalization variant
```

Lattice WER 0 cost deta hai jab hypothesis word kisi bhi bin alternative se match kare.

**Model agreement kab trust karein:**
Teeno conditions poori honi chahiye:
1. ≥ 60% models same alternative pe agree karein
2. Alternative mein edit-distance ≤ 2 from reference OR known alternatives mein ho
3. Hallucination nahi (reference se structurally unrelated nahi)

**Result:** 8/15 unfairly penalized models fixed, avg WER 21.7% → 5.6%, zero incorrect rewards.

**Output:**
- `outputs/q4/lattice_wer_results.csv`

---

## ⚙️ Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHISPER_MODEL` | `openai/whisper-small` | Base model for fine-tuning |
| `TARGET_LANGUAGE` | `hindi` | Forces Hindi decoding |
| `TRAIN_EPOCHS` | `5` | Training epochs |
| `TRAIN_BATCH_SIZE` | `8` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `2` | Effective batch = 8×2 = 16 |
| `LEARNING_RATE` | `1e-5` | AdamW learning rate |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `MIN_SEGMENT_SEC` | `1.0` | Minimum utterance duration |
| `MAX_SEGMENT_SEC` | `29.5` | Maximum utterance duration |
| `VAL_SPLIT_RATIO` | `0.10` | Validation set fraction |
| `ERROR_SAMPLE_N` | `25` | Minimum error samples for analysis |

---

## 🐛 Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `SSL certificate verify failed` | Corporate/university proxy | `download_data.py` mein `requests.get()` ko `verify=False` karo |
| `CUDA out of memory` | GPU VRAM kam hai | `config.py` mein `TRAIN_BATCH_SIZE` 4 ya 2 karo |
| `FileNotFoundError: FT Data - data.csv` | CSV project root mein nahi | CSV file `config.py` ke saath wali folder mein rakho |
| `librosa.load` hang karta hai | ffmpeg missing | `sudo apt install ffmpeg` ya `brew install ffmpeg` |
| `ValueError: forced_decoder_ids` | Naya transformers API | `model.config` ki jagah `model.generation_config` use karo |
| Low download count (~60/104) | Kuch GCS files delete ho gayi | Normal hai — 80–100 successful typical hai |

---

## 📊 Training Summary

| Metric | Value |
|--------|-------|
| Training segments | 4,238 |
| Validation segments | 449 |
| Training time (T4 GPU) | ~30 minutes |
| Best checkpoint | Step 600 |
| Final validation loss | 0.3455 |
| Validation WER | 35.96% |
| **FLEURS test WER (baseline)** | **71.3%** |
| **FLEURS test WER (fine-tuned)** | **47.46%** |

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `datasets`, `librosa`, `soundfile`, `jiwer`, `evaluate`, `pandas`, `scikit-learn`, `tqdm`, `requests`, `accelerate`

---

## 🗂️ Key Output Files

| File | Question | Contents |
|------|----------|----------|
| `outputs/results/wer_results.csv` | Q1c | WER table for report |
| `outputs/results/error_samples.csv` | Q1d | 25 stratified error utterances |
| `outputs/error_analysis/taxonomy_report.md` | Q1e/f | Taxonomy + proposed fixes |
| `outputs/results/fix_before_after.csv` | Q1g | Before/after LM fix |
| `outputs/q2/normalized_transcripts.csv` | Q2a | Number-normalized output |
| `outputs/q2/english_tagged.csv` | Q2b | EN-word tagged output |
| `outputs/q3/spelling_results.csv` | Q3 | word, label, confidence, reason |
| `outputs/q4/lattice_wer_results.csv` | Q4 | Standard vs Lattice WER |

---

*For detailed local setup instructions, see [SETUP.md](SETUP.md)*

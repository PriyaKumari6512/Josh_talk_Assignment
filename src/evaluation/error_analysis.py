# src/evaluation/error_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Systematic error analysis of the fine-tuned model.
#
# Assignment requirements:
#   d) Sample at least 25 utterances where the model still makes errors.
#      Describe your sampling strategy.  Do NOT cherry-pick.
#   e) Build an error taxonomy.  For each category: reference, model output,
#      and your reasoning about the cause.
#   f) For the top 3 most frequent error types, propose a specific fix.
#
# Sampling strategy:
#   We use STRATIFIED SAMPLING BY CER (Character Error Rate) bucket:
#     - Low errors  : CER in (0, 0.30]   → sample 8
#     - Medium errors: CER in (0.30, 0.70] → sample 9
#     - High errors : CER > 0.70          → sample 8
#   This gives a representative sample across error severity, avoiding the
#   bias of picking only the worst (cherry-picking in reverse) or only the
#   easiest mistakes.
#
# Error taxonomy categories (emerging from the data):
#   1. Homophone Confusion    – similar-sounding words swapped
#   2. Code-switch Error      – Hindi spoken with English words, model transcribes wrong script
#   3. Compound Word Splitting – single word split into two or joined incorrectly
#   4. Numerical Expression   – numbers in words vs digits
#   5. Deletion Errors        – short function words (है, ने, को) dropped
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

try:
    import jiwer
except ImportError:
    raise ImportError("pip install jiwer")

PREDICTIONS_CSV = os.path.join(config.RESULTS_DIR, "all_predictions.csv")


# ── CER helper ─────────────────────────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    Uses jiwer with character-level transforms.
    """
    if not reference:
        return 1.0
    # jiwer's cer() is not built-in; compute manually using wer on char tokens
    ref_chars = " ".join(list(reference.replace(" ", "_")))
    hyp_chars = " ".join(list(hypothesis.replace(" ", "_")))
    try:
        cer = jiwer.wer(ref_chars, hyp_chars)
    except Exception:
        cer = 1.0
    return min(cer, 1.0)


def compute_wer_single(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0
    try:
        return min(jiwer.wer(reference, hypothesis), 1.0)
    except Exception:
        return 1.0


# ── Taxonomy classification ────────────────────────────────────────────────────

CATEGORY_PATTERNS = {
    # Code-switch error: reference has Roman/ASCII letters, hypothesis has Devanagari
    "Code-switch Error": lambda ref, hyp: (
        bool(re.search(r'[a-zA-Z]', ref)) or bool(re.search(r'[a-zA-Z]', hyp))
    ),
    # Numerical expression: reference or hypothesis has digits or Hindi number words
    "Numerical Expression Error": lambda ref, hyp: (
        bool(re.search(r'\d', ref + hyp)) or
        bool(re.search(r'(एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस|सौ|हज़ार|लाख)', ref + hyp))
    ),
}

FUNCTION_WORDS = {"है", "हैं", "ने", "को", "का", "की", "के", "में", "पर", "से", "यह", "वह", "और", "भी", "तो", "ही", "न", "नहीं"}

def classify_error(ref_words: list, hyp_words: list, ref: str, hyp: str) -> str:
    """
    Assign the primary error category for one utterance.
    Returns the name of the category.
    """
    # Code-switch
    if bool(re.search(r'[a-zA-Z]', ref + hyp)):
        return "Code-switch Error"

    # Numbers
    number_words = r'(एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस|सौ|हज़ार|लाख|\d+)'

    if re.search(number_words, ref) or re.search(number_words, hyp):
    # only if actual numeric meaning changed
         if any(num in ref and num not in hyp for num in re.findall(number_words, ref)):
            return "Numerical Expression Error"
    # Deletion of function words: ref has function word, hyp doesn't
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    deleted_func = ref_set.intersection(FUNCTION_WORDS) - hyp_set
    if deleted_func:
        return "Function Word Deletion"

    # Substitution (homophone confusion): length similar but words differ
    if len(ref_words) == len(hyp_words):
        diff = sum(1 for r, h in zip(ref_words, hyp_words) if r != h)
        if diff / max(len(ref_words), 1) < 0.5:
            return "Homophone / Near-Homophone Confusion"

    # Compound word error: different number of words (splitting or merging)
    len_diff = abs(len(ref_words) - len(hyp_words))
    if len_diff >= 1:
        return "Compound Word Split / Merge"

    return "Other Substitution"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 6 – Error Analysis")
    print("=" * 60)

    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Predictions CSV not found: {PREDICTIONS_CSV}\n"
            "Run src/evaluation/evaluate_wer.py first."
        )

    # ── Load predictions ───────────────────────────────────────────────────────
    rows = []
    with open(PREDICTIONS_CSV, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    print(f"\nTotal test utterances: {len(rows)}")

    # ── Filter to utterances where fine-tuned model makes errors ──────────────
    error_rows = []
    for row in rows:
        ref = row["reference"].strip()
        hyp = row["finetuned"].strip()
        if not ref or not hyp:
            continue
        wer = compute_wer_single(ref, hyp)
        cer = compute_cer(ref, hyp)
        if wer > 0:   # any error
            row["wer"] = round(wer, 4)
            row["cer"] = round(cer, 4)
            error_rows.append(row)

    print(f"Utterances with errors: {len(error_rows)}")

    # ── Stratified sampling by CER bucket ─────────────────────────────────────
    low_bucket    = [r for r in error_rows if float(r["cer"]) <= 0.30]
    medium_bucket = [r for r in error_rows if 0.30 < float(r["cer"]) <= 0.70]
    high_bucket   = [r for r in error_rows if float(r["cer"]) > 0.70]

    random.seed(42)   # reproducibility
    sample_low    = random.sample(low_bucket,    min(8, len(low_bucket)))
    sample_medium = random.sample(medium_bucket, min(9, len(medium_bucket)))
    sample_high   = random.sample(high_bucket,   min(8, len(high_bucket)))

    sampled = sample_low + sample_medium + sample_high
    print(f"\nSampled utterances: {len(sampled)}")
    print(f"  Low CER    (≤0.30) : {len(sample_low)}")
    print(f"  Medium CER (0.30–0.70): {len(sample_medium)}")
    print(f"  High CER   (>0.70) : {len(sample_high)}")

    # ── Save error samples CSV ─────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    error_csv_path = config.ERROR_CSV
    with open(error_csv_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["segment_id", "reference", "pred_finetuned",
                      "pred_baseline", "wer", "cer", "cer_bucket"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sampled:
            bucket = ("low" if float(r["cer"]) <= 0.30 else
                      "medium" if float(r["cer"]) <= 0.70 else "high")
            writer.writerow({
                "segment_id": os.path.basename(r["audio_path"]),
                "reference":      r["reference"],
                "pred_finetuned":      r["finetuned"],
                "pred_baseline":       r["baseline"],
                "wer":            r["wer"],
                "cer":            r["cer"],
                "cer_bucket":     bucket,
            })
    print(f"\n✅ Error samples saved: {error_csv_path}")

    # ── Build taxonomy ─────────────────────────────────────────────────────────
    taxonomy = defaultdict(list)
    for r in sampled:
        ref_words = r["reference"].split()
        hyp_words = r["finetuned"].split()
        category  = classify_error(ref_words, hyp_words, r["reference"], r["finetuned"])
        taxonomy[category].append(r)

    print("\n--- Error Taxonomy Summary ---")
    for cat, items in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(items)} examples")

    # ── Write taxonomy report ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.TAXONOMY_REPORT), exist_ok=True)
    with open(config.TAXONOMY_REPORT, "w", encoding="utf-8") as fh:
        fh.write("# Error Taxonomy Report – Whisper-small (Fine-tuned Hindi)\n\n")
        fh.write("## Sampling Strategy\n\n")
        fh.write(
            "Errors were stratified into three CER buckets and sampled proportionally:\n"
            "- **Low** (CER ≤ 0.30): 8 samples\n"
            "- **Medium** (CER 0.30–0.70): 9 samples\n"
            "- **High** (CER > 0.70): 8 samples\n\n"
            "> Random seed = 42 for reproducibility. "
            "This avoids cherry-picking by ensuring representation across all error severity levels.\n\n"
        )
        fh.write("## Taxonomy\n\n")

        for cat, items in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
            fh.write(f"### {cat} ({len(items)} examples)\n\n")
            fh.write("| # | Reference | Model Output | Cause |\n")
            fh.write("|---|-----------|--------------|-------|\n")
            for i, r in enumerate(items[:5], 1):   # show up to 5 per category
                ref = r["reference"]
                hyp = r["finetuned"]
                cause = get_cause_explanation(cat, ref, hyp)
                fh.write(f"| {i} | {ref} | {hyp} | {cause} |\n")
            fh.write("\n")

        # ── Top 3 fixes ───────────────────────────────────────────────────────
        top3 = sorted(taxonomy.items(), key=lambda x: -len(x[1]))[:3]
        fh.write("## Proposed Fixes for Top 3 Error Categories\n\n")
        fixes = get_top3_fixes([cat for cat, _ in top3])
        for cat, fix in fixes.items():
            fh.write(f"### {cat}\n\n{fix}\n\n")

    print(f"\n✅ Taxonomy report saved: {config.TAXONOMY_REPORT}")
    print("=" * 60)


def get_cause_explanation(category: str, ref: str, hyp: str) -> str:
    causes = {
        "Code-switch Error":
            "English word in Hindi speech; model transcribes in wrong script or omits it",
        "Numerical Expression Error":
            "Hindi number word not converted to digit or vice versa by the model",
        "Function Word Deletion":
            "Short function word acoustically weak; model drops it during decoding",
        "Homophone / Near-Homophone Confusion":
            "Words sound identical or very similar; model picks wrong lexical form",
        "Compound Word Split / Merge":
            "Sandhi or compound boundary not recognized; model splits or merges incorrectly",
        "Other Substitution":
            "Acoustic ambiguity or rare vocabulary not well-represented in training data",
    }
    return causes.get(category, "Unknown")


def get_top3_fixes(categories: list[str]) -> dict[str, str]:
    fix_db = {
        "Code-switch Error": (
            "**Fix: Script-normalised data augmentation**\n\n"
            "The model struggles because training data has English words in Devanagari "
            "(per the transcription guideline) but the pretrained Whisper was trained with "
            "English words in Roman script.\n\n"
            "1. Identify all Devanagari-transliterated English words in the training transcripts "
            "   (e.g., using a transliteration dictionary like `indic-transliteration`).\n"
            "2. For those utterances, train the model with BOTH the Devanagari form AND a "
            "   Roman-script alternative as parallel labels (multi-reference training).\n"
            "3. Add code-switched Hindi-English sentences from the CALCS / LINCE datasets "
            "   to the training set to expose the model to mixed-script input.\n\n"
            "*Why not just collect more data*: More monolingual Hindi data won't help. "
            "The model needs code-switched examples specifically."
        ),
        "Function Word Deletion": (
            "**Fix: Post-correction language model re-scoring**\n\n"
            "Function words (है, ने, को) are acoustically short and often inaudible in "
            "conversational speech.  Collecting more audio won't fix this because the "
            "signal is genuinely weak.\n\n"
            "1. Train a small Hindi n-gram or neural language model on a large Hindi text "
            "   corpus (e.g., CC-100 Hindi, Oscar Hindi).\n"
            "2. At inference time, use beam search with LM shallow fusion: "
            "   `final_score = whisper_score + alpha * lm_score`. The LM will heavily "
            "   penalise grammatically incomplete sequences missing postpositions.\n"
            "3. Alternatively, use a Hindi grammar checker as a post-processing step to "
            "   insert likely missing postpositions.\n\n"
            "*Why not just fine-tune more*: The model will never hallucinate acoustically "
            "absent words just from more audio fine-tuning."
        ),
        "Homophone / Near-Homophone Confusion": (
            "**Fix: Phoneme-aware contrastive fine-tuning**\n\n"
            "Hindi has many near-homophone pairs (e.g., मेल/मैल, कर/कड़, तेल/टेल). "
            "The model confuses them because their spectral fingerprints overlap.\n\n"
            "1. Create a confusion matrix of the most-swapped word pairs from the error "
            "   analysis output (outputs/results/error_samples.csv).\n"
            "2. For each confused pair, collect or generate minimal pair audio examples "
            "   (a sentence with 'मेल' and the same sentence with 'मैल').\n"
            "3. Fine-tune with a **contrastive loss** term that increases the distance "
            "   between the embeddings of confused phonemes in the encoder.\n"
            "4. Short-term fix: add a pronunciation lexicon to constrain beam search "
            "   decoding (force phonetically impossible paths to have -inf score).\n"
        ),
        "Numerical Expression Error": (
            "**Fix: Number normalisation post-processing layer**\n\n"
            "This is better solved as a text normalisation step (see Question 2) "
            "rather than by retraining the acoustic model.\n\n"
            "1. Build a rule-based Hindi number word → digit converter "
            "   (implemented fully in src/data/number_normalizer.py for Q2).\n"
            "2. Run it as a post-processing step after Whisper inference.\n"
            "3. This decouples the ASR model (predicts what was SAID) from "
            "   the normalisation layer (converts to canonical written form).\n"
        ),
        "Compound Word Split / Merge": (
            "**Fix: Sentencepiece vocabulary tuning**\n\n"
            "Whisper uses a byte-pair encoding (BPE) tokeniser trained on multilingual "
            "data.  Hindi compound words (e.g., 'कार्यालय', 'महाविद्यालय') are "
            "over-segmented into subwords, causing the decoder to output them as "
            "separate tokens that it then mis-joins.\n\n"
            "1. Train a domain-specific BPE vocabulary on your Hindi corpus with "
            "   `sentencepiece` (vocab size ~8000).\n"
            "2. Initialise the Whisper tokeniser with this domain vocabulary as a "
            "   custom extension, keeping the original multilingual vocab as fallback.\n"
            "3. Fine-tune the embedding layer (only) for 1 epoch on the domain vocab "
            "   before the full fine-tune.\n"
        ),
        "Other Substitution": (
            "**Fix: Data augmentation with rare-word emphasis**\n\n"
            "1. Identify words that appear fewer than 5 times in the training manifests.\n"
            "2. Use TTS (e.g., Google TTS in Hindi) to synthesise additional utterances "
            "   containing these rare words.\n"
            "3. Add synthesised audio to the training set, clearly labelled as synthetic "
            "   to allow curriculum-learning (train on real data first, then mix).\n"
        ),
    }
    return {cat: fix_db.get(cat, "No specific fix documented yet.") for cat in categories}


if __name__ == "__main__":
    main()

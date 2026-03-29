# src/postprocessing/q4_lattice_wer.py
# =============================================================================
# Q4 — Lattice-based WER Evaluation
#
# Problem with normal WER:
#   Reference: "उसने चौदह किताबें खरीदीं"
#   Model A:   "उसने 14 किताबें खरीदीं"      → Normal WER = 0.25 (penalized!)
#   Model B:   "उसने चौदह किताबे खरीदी"       → Normal WER = 0.50 (penalized!)
#   Both are semantically correct but get penalized unfairly.
#
# Solution: Lattice
#   A lattice replaces the flat reference string with a sequential list of
#   "bins". Each bin holds ALL valid alternatives at that alignment position.
#
#   Lattice: [["उसने"], ["चौदह","14"], ["किताबें","किताबे","पुस्तकें"], ["खरीदीं","खरीदी"]]
#
#   Now Model A's "14" matches bin 2 → WER = 0.0 ✅
#
# Alignment unit chosen: WORD level
#   Justification:
#   - Subword would over-penalize partial matches in agglutinative Hindi
#   - Phrase level would miss valid single-word substitutions
#   - Word level is the standard WER unit and works best for Hindi
# =============================================================================

import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "q4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1: LATTICE CONSTRUCTION
# =============================================================================

# Known valid alternatives for Hindi words
# (number words ↔ digits, spelling variants, synonyms)
VALID_ALTERNATIVES = {
    # Number word ↔ digit
    "एक":   ["1","एक"],
    "दो":   ["2","दो"],
    "तीन":  ["3","तीन"],
    "चार":  ["4","चार"],
    "पांच": ["5","पांच","पाँच"],
    "पाँच": ["5","पांच","पाँच"],
    "छह":   ["6","छह","छः"],
    "सात":  ["7","सात"],
    "आठ":   ["8","आठ"],
    "नौ":   ["9","नौ"],
    "दस":   ["10","दस"],
    "बीस":  ["20","बीस"],
    "सौ":   ["100","सौ"],
    "हज़ार": ["1000","हज़ार","हजार"],
    "हजार": ["1000","हज़ार","हजार"],
    "चौदह": ["14","चौदह"],
    "पंद्रह":["15","पंद्रह"],

    # Common Hindi spelling variants
    "किताबें":  ["किताबें","किताबे","पुस्तकें","पुस्तके"],
    "किताबे":   ["किताबें","किताबे","पुस्तकें"],
    "खरीदीं":   ["खरीदीं","खरीदी","ख़रीदीं"],
    "खरीदी":    ["खरीदीं","खरीदी"],
    "नहीं":     ["नहीं","नही"],
    "नही":      ["नहीं","नही"],
    "हां":      ["हां","हाँ","हा"],
    "हाँ":      ["हां","हाँ","हा"],
    "यहां":     ["यहां","यहाँ"],
    "यहाँ":     ["यहां","यहाँ"],
    "वहां":     ["वहां","वहाँ"],
    "वहाँ":     ["वहां","वहाँ"],
    "कहां":     ["कहां","कहाँ"],
    "कहाँ":     ["कहां","कहाँ"],
    "क्यों":    ["क्यों","क्यो"],
    "लेकिन":    ["लेकिन","लेकिन","परंतु","मगर"],
    "मैं":      ["मैं","में"],   # common ASR confusion
    "ज़्यादा":  ["ज़्यादा","ज्यादा","ज्यादा"],
    "ज्यादा":   ["ज़्यादा","ज्यादा"],

    # Digits ↔ words (reverse mapping)
    "14":   ["14","चौदह"],
    "15":   ["15","पंद्रह"],
    "1":    ["1","एक"],
    "2":    ["2","दो"],
    "100":  ["100","सौ"],
    "1000": ["1000","हज़ार","हजार"],
}


def get_alternatives(word: str) -> List[str]:
    """Get all valid alternatives for a word."""
    word_clean = word.strip().lower()
    alts = VALID_ALTERNATIVES.get(word, [word])
    alts = VALID_ALTERNATIVES.get(word_clean, alts)
    if word not in alts:
        alts = [word] + [a for a in alts if a != word]
    return list(set(alts))


def word_align(ref: List[str], hyp: List[str]) -> List[Tuple]:
    """
    Align hypothesis to reference using dynamic programming (edit distance).
    Returns list of (ref_word_or_None, hyp_word_or_None, operation)
    Operations: MATCH, SUBSTITUTE, INSERT, DELETE
    """
    m, n = len(ref), len(hyp)

    # DP table
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # substitution

    # Backtrack
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            alignment.append((ref[i-1], hyp[j-1], "MATCH"))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((ref[i-1], hyp[j-1], "SUBSTITUTE"))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append((ref[i-1], None, "DELETE"))
            i -= 1
        else:
            alignment.append((None, hyp[j-1], "INSERT"))
            j -= 1

    return list(reversed(alignment))


def build_lattice(
    reference: str,
    model_outputs: Dict[str, str],
    agreement_threshold: float = 0.6
) -> List[List[str]]:
    """
    Build a lattice from reference + multiple model outputs.

    Algorithm:
    1. Tokenize all inputs into word lists
    2. Align each model output to the reference
    3. For each alignment position, collect all valid alternatives
    4. If majority of models agree on something different from reference
       AND that alternative is phonetically/semantically valid → add to bin
    5. Trust model agreement over reference when:
       - >= agreement_threshold fraction of models agree
       - The agreed-upon form is in VALID_ALTERNATIVES

    Args:
        reference        : human reference string
        model_outputs    : dict of {model_name: transcription_string}
        agreement_threshold: fraction of models that must agree to override ref

    Returns:
        lattice: list of bins, each bin is list of valid strings for that position
    """
    ref_words = reference.split()
    n_models  = len(model_outputs)

    # Align each model to reference
    alignments = {}
    for model_name, hyp in model_outputs.items():
        hyp_words = hyp.split()
        alignments[model_name] = word_align(ref_words, hyp_words)

    # Build lattice bin by bin
    lattice = []

    for i, ref_word in enumerate(ref_words):
        bin_alts = set(get_alternatives(ref_word))

        # Collect what models produced at this position
        model_votes = {}
        for model_name, alignment in alignments.items():
            # Find alignment entry for position i
            ref_pos = 0
            for (rw, hw, op) in alignment:
                if rw == ref_word and ref_pos == 0:
                    if hw is not None:
                        model_votes[model_name] = hw
                    ref_pos += 1
                    break

        # Count model votes
        vote_counter = {}
        for model_name, voted_word in model_votes.items():
            vote_counter[voted_word] = vote_counter.get(voted_word, 0) + 1

        # Trust model agreement: if >= threshold models agree on X
        # and X is a valid alternative → add X to bin
        for word, count in vote_counter.items():
            if count / n_models >= agreement_threshold:
                # Validate: must be phonetically/semantically related
                ref_alternatives = get_alternatives(ref_word)
                if (word in ref_alternatives or
                    word in VALID_ALTERNATIVES or
                    edit_distance_simple(word, ref_word) <= 2):
                    bin_alts.add(word)

        lattice.append(sorted(bin_alts))

    return lattice


def edit_distance_simple(s1: str, s2: str) -> int:
    """Fast edit distance for short strings."""
    if s1 == s2: return 0
    m, n = len(s1), len(s2)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            temp = dp[j]
            dp[j] = prev if s1[i-1]==s2[j-1] else 1+min(prev,dp[j],dp[j-1])
            prev = temp
    return dp[n]


# =============================================================================
# PART 2: LATTICE WER COMPUTATION
# =============================================================================

def compute_lattice_wer(hypothesis: str, lattice: List[List[str]]) -> Dict:
    """
    Compute WER against a lattice instead of a flat reference string.

    For each position in the lattice, a hypothesis word MATCHES if it
    appears in ANY bin alternative — not just the exact reference word.

    This is fair because all bin entries are valid transcriptions.

    Returns:
        dict with wer, substitutions, deletions, insertions, matches
    """
    hyp_words = hypothesis.split()
    n_ref     = len(lattice)
    n_hyp     = len(hyp_words)

    # DP over lattice positions × hypothesis positions
    # Cost: 0 if hyp_word is in lattice[i], else 1
    dp = [[0]*(n_hyp+1) for _ in range(n_ref+1)]
    for i in range(n_ref+1): dp[i][0] = i
    for j in range(n_hyp+1): dp[0][j] = j

    for i in range(1, n_ref+1):
        for j in range(1, n_hyp+1):
            # Match: hypothesis word is in this lattice bin
            in_bin = hyp_words[j-1] in lattice[i-1]
            if in_bin:
                dp[i][j] = dp[i-1][j-1]           # match → no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion (ref word skipped)
                    dp[i][j-1],    # insertion (extra hyp word)
                    dp[i-1][j-1]   # substitution
                )

    # Count operation types via backtrack
    subs = dels = ins = matches = 0
    i, j = n_ref, n_hyp
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp_words[j-1] in lattice[i-1]:
            matches += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1

    errors = subs + dels + ins
    wer    = errors / max(n_ref, 1)

    return {
        "wer":           round(wer, 4),
        "errors":        errors,
        "substitutions": subs,
        "deletions":     dels,
        "insertions":    ins,
        "matches":       matches,
        "ref_length":    n_ref,
    }


def compute_standard_wer(hypothesis: str, reference: str) -> float:
    """Standard WER against flat reference string."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            temp = dp[j]
            dp[j] = prev if ref_words[i-1]==hyp_words[j-1] else 1+min(prev,dp[j],dp[j-1])
            prev = temp
    return round(dp[m] / max(n, 1), 4)


# =============================================================================
# PART 3: DEMONSTRATION WITH EXAMPLES
# =============================================================================

DEMO_EXAMPLES = [
    {
        "description": "Number word vs digit — both valid",
        "reference": "उसने चौदह किताबें खरीदीं",
        "models": {
            "Model_A": "उसने 14 किताबें खरीदीं",      # 14 = चौदह, valid
            "Model_B": "उसने चौदह किताबे खरीदी",       # spelling variant, valid
            "Model_C": "उसने चौदह किताबें खरीदीं",     # exact match
            "Model_D": "उसने तीन किताबें खरीदीं",       # genuinely wrong (तीन≠चौदह)
            "Model_E": "उसने चौदह पुस्तकें खरीदीं",    # synonym, valid
        }
    },
    {
        "description": "Spelling variants and reference error",
        "reference": "मुझे यहाँ आना था",
        "models": {
            "Model_A": "मुझे यहां आना था",   # यहाँ/यहां both valid
            "Model_B": "मुझे यहाँ आना था",   # exact
            "Model_C": "मुझे यहां आना था",   # variant
            "Model_D": "मुझे यहां आना था",   # variant
            "Model_E": "मुझे वहां आना था",   # genuinely wrong (वहां≠यहाँ)
        }
    },
    {
        "description": "नहीं vs नही — common ASR variation",
        "reference": "मैं नहीं जाऊंगा",
        "models": {
            "Model_A": "मैं नही जाऊंगा",    # valid variant
            "Model_B": "मैं नहीं जाऊंगा",   # exact
            "Model_C": "मैं नही जाऊंगा",    # valid variant
            "Model_D": "मैं नहीं जाऊंगा",   # exact
            "Model_E": "मैं कभी जाऊंगा",    # wrong (कभी≠नहीं)
        }
    },
]


def run_demo():
    print("=" * 65)
    print("Q4 — Lattice-based WER: Theory + Implementation")
    print("=" * 65)

    print("""
ALIGNMENT UNIT: WORD
Justification:
  - Standard WER unit for Hindi ASR evaluation
  - Subword would over-penalize partial matches in agglutinative Hindi
    (e.g. "किताबें" vs "किताबे" — 1 word error, not 3 subword errors)
  - Phrase-level too coarse — misses valid single-word substitutions
  - Word level balances granularity and interpretability
    """)

    all_results = []

    for example in DEMO_EXAMPLES:
        ref     = example["reference"]
        models  = example["models"]
        desc    = example["description"]

        print(f"\n{'─'*65}")
        print(f"Example: {desc}")
        print(f"Reference : {ref}")

        # Build lattice
        lattice = build_lattice(ref, models, agreement_threshold=0.6)

        print(f"\nLattice built:")
        for i, bin_list in enumerate(lattice):
            print(f"  Bin {i+1}: {bin_list}")

        print(f"\nWER Comparison (Standard vs Lattice):")
        print(f"  {'Model':<12} {'Hypothesis':<40} {'Std WER':>8} {'Lat WER':>8} {'Δ':>8} {'Fair?':>6}")
        print(f"  {'─'*12} {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

        for model_name, hyp in models.items():
            std_wer = compute_standard_wer(hyp, ref)
            lat     = compute_lattice_wer(hyp, lattice)
            lat_wer = lat["wer"]
            delta   = lat_wer - std_wer
            fair    = "✅" if delta < 0 else ("—" if delta == 0 else "⚠")
            print(f"  {model_name:<12} {hyp:<40} {std_wer:>8.3f} {lat_wer:>8.3f} {delta:>+8.3f} {fair:>6}")

            all_results.append({
                "example":    desc,
                "model":      model_name,
                "reference":  ref,
                "hypothesis": hyp,
                "std_wer":    std_wer,
                "lat_wer":    lat_wer,
                "improved":   delta < -0.001,
            })

    # Summary
    print(f"\n{'='*65}")
    print("Summary across all examples")
    print(f"{'='*65}")
    improved  = sum(1 for r in all_results if r["improved"])
    unchanged = sum(1 for r in all_results if not r["improved"] and r["lat_wer"] <= r["std_wer"])
    regressed = sum(1 for r in all_results if r["lat_wer"] > r["std_wer"])
    print(f"  Models fairly corrected (WER reduced) : {improved}/{len(all_results)}")
    print(f"  Models unchanged (already fair)        : {unchanged}/{len(all_results)}")
    print(f"  Models regressed (WER increased)       : {regressed}/{len(all_results)}")

    avg_std = sum(r["std_wer"] for r in all_results) / len(all_results)
    avg_lat = sum(r["lat_wer"] for r in all_results) / len(all_results)
    print(f"\n  Average Standard WER : {avg_std:.3f} ({avg_std*100:.1f}%)")
    print(f"  Average Lattice  WER : {avg_lat:.3f} ({avg_lat*100:.1f}%)")
    print(f"  Average improvement  : {(avg_lat-avg_std)*100:+.1f}%")

    # Save results
    import csv
    out_csv = OUTPUT_DIR / "lattice_wer_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    out_json = OUTPUT_DIR / "lattice_explanation.json"
    explanation = {
        "approach": "Word-level lattice WER",
        "alignment_unit": "word",
        "why_word_level": [
            "Standard WER unit — comparable to published results",
            "Hindi is moderately agglutinative — subword over-penalizes",
            "Phrase level too coarse for single-word valid variants",
        ],
        "lattice_construction": [
            "1. Tokenize reference and all model outputs into words",
            "2. Align each model output to reference via edit distance DP",
            "3. For each reference word position, collect all model outputs",
            "4. Add VALID_ALTERNATIVES (number variants, spelling variants, synonyms)",
            "5. If >= 60% models agree on X AND X is phonetically valid → add to bin",
            "6. Result: each bin contains all valid transcriptions for that position",
        ],
        "when_to_trust_models_over_reference": [
            "Condition 1: >= 60% of models produce the same alternative",
            "Condition 2: The alternative is phonetically/semantically valid",
            "Condition 3: Edit distance to reference word <= 2 (likely variant)",
            "Example: If 4/5 models say '14' and reference says 'चौदह' → both valid",
        ],
        "how_lattice_wer_differs": [
            "Standard WER: penalty if hypothesis != exact reference word",
            "Lattice WER : no penalty if hypothesis word appears in ANY bin alternative",
            "Effect: models penalized for valid variants get WER reduction",
            "Models genuinely wrong (wrong meaning) remain penalized",
        ],
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(explanation, fh, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved: {out_csv}")
    print(f"✅ Explanation  : {out_json}")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
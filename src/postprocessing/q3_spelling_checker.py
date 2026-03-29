# src/postprocessing/q3_spelling_checker.py
# =============================================================================
# Q3 — Hindi Spelling Error Detection (FINAL WORKING VERSION)
# =============================================================================

import os
import re
import json
import csv
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILES = [
    BASE_DIR / "data" / "train.jsonl",
    BASE_DIR / "data" / "val.jsonl"
]

OUTPUT_DIR = BASE_DIR / "outputs" / "q3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_LIST_OUTPUT = OUTPUT_DIR / "spelling_results.csv"
STATS_OUTPUT = OUTPUT_DIR / "spelling_stats.json"
LOW_CONF_REVIEW = OUTPUT_DIR / "low_confidence_review.csv"

# ── Known correct words (expanded) ─────────────────────────────────────────────
KNOWN_CORRECT = {
    "है","हैं","था","थी","थे","में","को","के","की","का","से","पर","और","या",
    "नहीं","मैं","हम","आप","यह","वह","एक","दो","तीन","क्या","कौन","सब","कुछ",
    "बहुत","अब","फिर","पहले","बाद","साथ","लेकिन","इसलिए",
    "दिन","रात","समय","साल","घर","काम","लोग","बात","जगह",
    "करना","होना","जाना","आना","देना","लेना","बोलना","देखना",
    "अच्छा","बुरा","बड़ा","छोटा","नया","पुराना","सही","गलत",
    "जंगल","रास्ता","रास्ते","पहली","बार","वहां","होते","किया","देखा"
}

ENGLISH_LOANWORDS_DEVANAGARI = {
    "स्कूल","कॉलेज","फोन","कंप्यूटर","मोबाइल","जॉब","ऑफिस","ट्रेन",
    "बस","टीचर","वीडियो","इंटरनेट","ऑनलाइन","ऐप","मीटिंग","प्रोजेक्ट"
}

# ── Pattern checks ─────────────────────────────────────────────────────────────
def has_invalid_pattern(word):
    if re.search(r'[\u093E-\u094C]{2,}', word):
        return True, "double matra"
    if word.startswith('\u094D'):
        return True, "invalid start"
    if re.search(r'(.)\1\1', word):
        return True, "triple char repetition"
    return False, ""

# ── Edit distance ──────────────────────────────────────────────────────────────
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n+1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def is_typo(word):
    for w in list(KNOWN_CORRECT)[:100]:  # limit for speed
        if abs(len(w) - len(word)) <= 2:
            if edit_distance(word, w) == 1:
                return True
    return False

# ── Classification ─────────────────────────────────────────────────────────────
def classify(word, freq):

    # Known correct
    if word in KNOWN_CORRECT:
        return "correct", "high", "known word"

    # English loanword
    if word in ENGLISH_LOANWORDS_DEVANAGARI:
        return "correct", "high", "loanword"

    # Invalid pattern
    invalid, reason = has_invalid_pattern(word)
    if invalid:
        return "incorrect", "high", reason

    # Typo detection
    if is_typo(word):
        return "incorrect", "high", "edit distance typo"

    # Short words
    if len(word) <= 2:
        return "correct", "high", "short word"

    # High frequency
    if freq >= 20:
        return "correct", "high", "high freq"

    if freq >= 5:
        return "correct", "medium", "medium freq"

    # Proper noun heuristic
    if freq <= 2 and len(word) >= 4:
        return "correct", "low", "possible name/rare word"

    # Default
    return "correct", "low", "uncertain"

# ── Load JSONL safely ─────────────────────────────────────────────────────────
def load_all_text():
    all_words = []

    for file in DATA_FILES:
        if not file.exists():
            print(f"⚠ Missing file: {file}")
            continue

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    words = re.findall(r'[\u0900-\u097F]+', text)
                    all_words.extend(words)
                except:
                    continue

    return all_words

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("Q3 — FINAL SPELLING CHECKER (IMPROVED)")
    print("="*60)

    words = load_all_text()

    if not words:
        print("❌ No data found")
        return

    vocab = Counter(words)

    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(vocab)}")

    results = []

    for word, freq in vocab.items():
        label, conf, reason = classify(word, freq)

        results.append({
            "word": word,
            "freq": freq,
            "label": label,
            "confidence": conf,
            "reason": reason
        })

    # Save CSV
    with open(WORD_LIST_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word","freq","label","confidence","reason"])
        writer.writeheader()
        writer.writerows(results)

    # Stats
    correct = sum(1 for r in results if r["label"] == "correct")
    incorrect = sum(1 for r in results if r["label"] == "incorrect")

    stats = {
        "total_unique": len(vocab),
        "correct": correct,
        "incorrect": incorrect
    }

    with open(STATS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Low confidence sample
    low_conf = [r for r in results if r["confidence"] == "low"][:50]

    with open(LOW_CONF_REVIEW, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word","freq","label","confidence","reason"])
        writer.writeheader()
        writer.writerows(low_conf)

    print("\n✅ DONE")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Saved: {WORD_LIST_OUTPUT}")

if __name__ == "__main__":
    main()
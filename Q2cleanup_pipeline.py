"""
Q2: ASR Cleanup Pipeline for Hindi Transcriptions
==================================================
Two operations:
  a) Number Normalization  — Hindi number words → digits
  b) English Word Detection — tag Devanagari-transliterated English words

Author: Josh Talks AI Researcher Intern Assignment
"""

import os
import re
import json
from pathlib import Path

# ─────────────────────────────────────────────
# PART A: NUMBER NORMALIZATION
# ─────────────────────────────────────────────

# Units: 0–19
UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}

# Tens: 20–99
TENS = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23,
    "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27,
    "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30, "इकतीस": 31,
    "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चौवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47,
    "अड़तालीस": 48, "उनचास": 49, "पचास": 50, "इक्यावन": 51,
    "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63,
    "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67,
    "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71,
    "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83,
    "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87,
    "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91,
    "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पंचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000,
    "हजार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
}

# Idioms/phrases where number words should NOT be converted
IDIOM_PATTERNS = [
    r"दो-चार",
    r"चार-पाँच",
    r"तीन-चार",
    r"दो-तीन",
    r"पाँच-दस",
    r"दस-बीस",
    r"एक-दो",
    r"एक न एक",
    r"दो टूक",
    r"तीन तिकड़म",
    r"नौ दो ग्यारह",
    r"एक से बढ़कर एक",
    r"सात समंदर",
    r"चार चाँद",
]

def is_idiom(text, match_start, match_end):
    """Check if a number match falls within a known idiom."""
    for pattern in IDIOM_PATTERNS:
        for m in re.finditer(pattern, text):
            if m.start() <= match_start and m.end() >= match_end:
                return True
    return False

def parse_number_sequence(tokens):
    """
    Convert a list of Hindi number word tokens to a single integer.
    Example: ['तीन', 'सौ', 'चौवन'] -> 354
    """
    result = 0
    current = 0

    for tok in tokens:
        if tok in UNITS:
            current += UNITS[tok]
        elif tok in TENS:
            current += TENS[tok]
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if mult == 100:
                current = (current if current > 0 else 1) * 100
            elif mult >= 1000:
                if current == 0:
                    current = 1
                result += current * mult
                current = 0

    result += current
    return result if result > 0 else None

def normalize_numbers(text):
    """
    Finds sequences of Hindi number words and replaces with digits.
    Skips idioms where conversion would be wrong.
    """
    all_num_keys = set(list(UNITS.keys()) + list(TENS.keys()) + list(MULTIPLIERS.keys()))
    words = text.split()

    word_positions = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        word_positions.append((start, start + len(word), word))
        pos = start + len(word)

    i = 0
    output_words = []

    while i < len(words):
        word = words[i]
        clean_word = word.strip(',.।?!')

        if clean_word in all_num_keys:
            span_start = word_positions[i][0]
            span_end = word_positions[i][1]

            if is_idiom(text, span_start, span_end):
                output_words.append(word)
                i += 1
                continue

            num_seq = [clean_word]
            punct_after = word[len(clean_word):]

            j = i + 1
            while j < len(words):
                next_clean = words[j].strip(',.।?!')
                if next_clean in all_num_keys:
                    num_seq.append(next_clean)
                    j += 1
                else:
                    break

            span_end_j = word_positions[min(j - 1, len(word_positions) - 1)][1]
            if is_idiom(text, span_start, span_end_j):
                output_words.extend(words[i:j])
                i = j
                continue

            num_val = parse_number_sequence(num_seq)
            if num_val is not None:
                output_words.append(str(num_val) + punct_after)
            else:
                output_words.extend(words[i:j])

            i = j
        else:
            output_words.append(word)
            i += 1

    return " ".join(output_words)

# ─────────────────────────────────────────────
# PART B: ENGLISH WORD DETECTION (Devanagari)
# ─────────────────────────────────────────────

ENGLISH_LOANWORDS_DEVANAGARI = {
    # Education
    "स्कूल", "कॉलेज", "क्लास", "टीचर", "प्रोफेसर", "एग्जाम", "रिजल्ट",
    "सिलेबस", "फीस", "एडमिशन", "सर्टिफिकेट", "डिग्री", "कोर्स", "सब्जेक्ट",

    # Technology
    "कंप्यूटर", "मोबाइल", "फोन", "इंटरनेट", "ऑनलाइन", "ऐप", "सॉफ्टवेयर",
    "वेबसाइट", "ईमेल", "व्हाट्सएप", "फेसबुक", "यूट्यूब", "इंस्टाग्राम",
    "लैपटॉप", "टैबलेट", "स्क्रीन", "कैमरा", "बैटरी", "चार्जर",

    # Work / Office
    "जॉब", "ऑफिस", "मैनेजर", "बॉस", "टीम", "प्रोजेक्ट", "मीटिंग",
    "इंटरव्यू", "रिज्यूमे", "सैलरी", "टारगेट", "फॉर्म", "रिपोर्ट",

    # Transport / Daily life
    "बस", "ट्रेन", "बाइक", "स्कूटी", "कार", "ट्रैफिक", "रोड",
    "होटल", "रेस्टोरेंट", "शॉप", "मॉल", "मार्केट",

    # Health
    "डॉक्टर", "हॉस्पिटल", "टेस्ट", "रिपोर्ट", "मेडिसिन", "ऑपरेशन",

    # Finance
    "बैंक", "लोन", "ईएमआई", "पेमेंट", "कैश", "अकाउंट",

    # Media / Entertainment
    "न्यूज़", "चैनल", "सीरियल", "मूवी", "सॉन्ग", "वीडियो", "गेम",

    # Common conversational loanwords
    "टाइम", "टाइप", "सिस्टम", "इशू", "प्रॉब्लम", "सॉल्यूशन",
    "स्टेशन", "प्लेटफॉर्म", "टिकट", "प्रेशर", "टेंशन",
}

EXCLUDE_FROM_ENGLISH = {
    "बस", "ज़रूर", "ज़रूरी", "ज़िंदगी", "ज़्यादा", "ज़माना", "ज़मीन",
    "ज़रा", "ज़ाहिर", "फ़र्क", "फ़िक्र", "फ़ायदा", "ख़ुद", "ख़ास",
    "ख़याल", "ख़ुशी", "ख़राब", "ग़लत", "ग़रीब", "चीज़", "चीज़ें",
    "साफ़", "आज़ाद", "मज़ा", "मज़बूत", "नज़र",
}

FOREIGN_PHONEMES = ['ऑ', 'ज़', 'फ़', 'क़', 'ग़', 'ख़']

def has_foreign_phoneme(word):
    return any(ph in word for ph in FOREIGN_PHONEMES)

def detect_english_words(text):
    """
    Identify Devanagari-transliterated English words in Hindi text.
    Returns tagged_text and list of detected words.
    """
    words = text.split()
    tagged_words = []
    english_words = []

    for word in words:
        clean = word.strip(',.।?!()')
        is_english = False

        if clean in EXCLUDE_FROM_ENGLISH:
            is_english = False
        elif clean in ENGLISH_LOANWORDS_DEVANAGARI:
            is_english = True
        elif has_foreign_phoneme(clean) and clean not in EXCLUDE_FROM_ENGLISH:
            is_english = True

        if is_english:
            punctuation = word[len(clean):]
            tagged_words.append(f"[EN]{clean}[/EN]{punctuation}")
            english_words.append(clean)
        else:
            tagged_words.append(word)

    tagged_text = " ".join(tagged_words)
    return tagged_text, english_words

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def process_transcript(text):
    normalized = normalize_numbers(text)
    tagged, eng_words = detect_english_words(normalized)
    return {
        "original": text,
        "number_normalized": normalized,
        "english_tagged": tagged,
        "english_words_found": eng_words,
    }

def load_segments_from_jsonl(base_dir):
    """Load transcript text from train.jsonl and val.jsonl."""
    all_segments = []

    for rel_path in ["data/train.jsonl", "data/val.jsonl"]:
        path = os.path.join(base_dir, rel_path)
        if not os.path.exists(path):
            print(f"Warning: file not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text", "").strip()
                    if text:
                        all_segments.append(text)
                except json.JSONDecodeError:
                    continue

    return all_segments

# ─────────────────────────────────────────────
# DEMO / EVALUATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE = "/data/datasets/josh/josh_talks_asr"

    all_segments = load_segments_from_jsonl(BASE)

    print("=" * 70)
    print("Q2a: NUMBER NORMALIZATION — Before/After Examples")
    print("=" * 70)

    if not all_segments:
        print("\nNo transcript segments found.")
        print("Checked files:")
        print(f"  {BASE}/data/train.jsonl")
        print(f"  {BASE}/data/val.jsonl")
        raise SystemExit(1)

    num_word_set = set(list(UNITS.keys()) + list(TENS.keys()) + list(MULTIPLIERS.keys()))
    shown = 0

    for seg in all_segments:
        words = seg.split()
        num_words_in_seg = [w.strip(',.।?!') for w in words if w.strip(',.।?!') in num_word_set]
        real_nums = [
            w for w in num_words_in_seg
            if w not in ('एक', 'दो') or num_words_in_seg.count(w) >= 2
        ]

        if not real_nums and not any(
            x in seg for x in ['तीन', 'पांच', 'पाँच', 'दस', 'बीस', 'सौ', 'हज़ार', 'हजार', 'पचास', 'तीस', 'चालीस']
        ):
            continue

        result = process_transcript(seg)
        if result["number_normalized"] != seg:
            print(f"\nExample {shown + 1}:")
            print(f"  BEFORE : {seg[:200]}")
            print(f"  AFTER  : {result['number_normalized'][:200]}")
            shown += 1
            if shown >= 6:
                break

    print("\n" + "=" * 70)
    print("Q2a: EDGE CASES — Idioms that should NOT be converted")
    print("=" * 70)

    edge_cases = [
        ("दो-चार बातें करनी थीं उससे", "दो-चार = 'a few' (idiom), not 2-4"),
        ("नौ दो ग्यारह हो गया", "नौ दो ग्यारह = ran away (idiom)"),
        ("एक न एक दिन सफलता मिलेगी", "एक न एक = one day or another (fixed expression)"),
        ("तीन सौ चौवन रुपये दिए", "तीन सौ चौवन = 354 (should convert)"),
        ("पच्चीस हज़ार की नौकरी मिली", "पच्चीस हज़ार = 25000 (should convert)"),
    ]

    for text, note in edge_cases:
        result = normalize_numbers(text)
        status = "✅ KEPT" if result == text else "🔢 CONVERTED"
        print(f"\n  Input   : {text}")
        print(f"  Output  : {result}")
        print(f"  Status  : {status} | Note: {note}")

    print("\n" + "=" * 70)
    print("Q2b: ENGLISH WORD DETECTION — Tagged Examples")
    print("=" * 70)

    shown2 = 0
    for seg in all_segments:
        if any(word in seg for word in ENGLISH_LOANWORDS_DEVANAGARI):
            result = process_transcript(seg)
            if result["english_words_found"]:
                print(f"\nExample {shown2 + 1}:")
                print(f"  ORIGINAL : {seg[:200]}")
                print(f"  TAGGED   : {result['english_tagged'][:220]}")
                print(f"  EN WORDS : {result['english_words_found']}")
                shown2 += 1
                if shown2 >= 6:
                    break

    print("\n" + "=" * 70)
    print("CORPUS-WIDE STATISTICS")
    print("=" * 70)

    total_segs = len(all_segments)
    num_changed = 0
    en_tagged = 0
    all_en_words = {}

    for seg in all_segments:
        r = process_transcript(seg)
        if r["number_normalized"] != seg:
            num_changed += 1
        if r["english_words_found"]:
            en_tagged += 1
            for w in r["english_words_found"]:
                all_en_words[w] = all_en_words.get(w, 0) + 1

    print(f"  Total segments       : {total_segs}")
    print(f"  Segments with number normalization applied : {num_changed} ({100 * num_changed / total_segs:.1f}%)")
    print(f"  Segments with English words tagged         : {en_tagged} ({100 * en_tagged / total_segs:.1f}%)")
    print("\n  Top 15 detected English loanwords:")

    for word, count in sorted(all_en_words.items(), key=lambda x: -x[1])[:15]:
        print(f"    '{word}' → {count} occurrences")

    print("\n✅ Pipeline complete.")
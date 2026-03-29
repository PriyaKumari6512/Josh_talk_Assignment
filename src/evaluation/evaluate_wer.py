import os
import sys
import json
import csv
import time
import librosa
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


FLEURS_MANIFEST = os.path.join(config.FLEURS_DIR, "test.jsonl")
PREDICTIONS_CSV = os.path.join(config.RESULTS_DIR, "all_predictions.csv")


# -------------------- HELPERS --------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_for_wer(text):
    return " ".join(text.strip().lower().split())


# -------------------- FIXED INFERENCE --------------------

def run_inference(rows, model_name_or_path, device="cpu", desc="Inference"):

    print(f"\n🚀 Loading model: {model_name_or_path}")

    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    model.to(device)
    model.eval()

    predictions = []

    for row in tqdm(rows, desc=desc):

        audio_path = row["audio_path"]

        if not os.path.exists(audio_path):
            predictions.append("")
            continue

        # Load audio
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)

        # Feature extraction
        inputs = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(device)

        # 🔥 FIXED GENERATION
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_new_tokens=128,      # ✅ FIX
                task="transcribe",       # ✅ modern way
                language="hi"            # ✅ Hindi
            )

        pred = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        predictions.append(pred.strip())

    return predictions


# -------------------- MAIN --------------------

def main():

    print("=" * 60)
    print("🔥 Evaluating Whisper on FLEURS Hindi")
    print("=" * 60)

    if not os.path.exists(FLEURS_MANIFEST):
        raise FileNotFoundError(f"❌ Missing: {FLEURS_MANIFEST}")

    rows = load_jsonl(FLEURS_MANIFEST)
    print(f"📊 Total samples: {len(rows)}")

    references = [normalize_for_wer(r["text"]) for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Using device: {device}")

    wer_metric = evaluate.load("wer")

    results = []

    # ---------------- BASELINE ----------------
    print("\n🔹 Running BASELINE model")

    t0 = time.time()
    preds_base = run_inference(
        rows,
        config.WHISPER_MODEL,
        device,
        desc="Baseline"
    )
    t1 = time.time()

    preds_base_norm = [normalize_for_wer(p) for p in preds_base]

    wer_base = wer_metric.compute(
        predictions=preds_base_norm,
        references=references
    )

    print(f"✅ Baseline WER: {wer_base*100:.2f}%")

    results.append({
        "Model": "whisper-small (pretrained)",
        "WER (%)": round(wer_base * 100, 2)
    })

    # ---------------- FINETUNED ----------------
    finetuned_path = os.path.join(config.MODEL_OUTPUT_DIR, "final")

    preds_ft_norm = [""] * len(rows)

    if os.path.exists(finetuned_path):

        print("\n🔹 Running FINE-TUNED model")

        t2 = time.time()
        preds_ft = run_inference(
            rows,
            finetuned_path,
            device,
            desc="Fine-tuned"
        )
        t3 = time.time()

        preds_ft_norm = [normalize_for_wer(p) for p in preds_ft]

        wer_ft = wer_metric.compute(
            predictions=preds_ft_norm,
            references=references
        )

        print(f"✅ Fine-tuned WER: {wer_ft*100:.2f}%")

        improvement = (wer_base - wer_ft) / wer_base * 100
        print(f"📈 Improvement: {improvement:.2f}%")

        results.append({
            "Model": "whisper-small (fine-tuned)",
            "WER (%)": round(wer_ft * 100, 2)
        })

    else:
        print("⚠ Fine-tuned model not found, skipping...")

    # ---------------- SAVE RESULTS ----------------
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(config.WER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "WER (%)"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📄 Results saved: {config.WER_CSV}")

    # ---------------- SAVE PREDICTIONS ----------------
    with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "reference", "baseline", "finetuned"]
        )
        writer.writeheader()

        for i, row in enumerate(rows):
            writer.writerow({
                "audio_path": row["audio_path"],
                "reference": references[i],
                "baseline": preds_base_norm[i],
                "finetuned": preds_ft_norm[i]
            })

    print(f"📄 Predictions saved: {PREDICTIONS_CSV}")

    print("\n🎯 DONE SUCCESSFULLY 🚀")


if __name__ == "__main__":
    main()
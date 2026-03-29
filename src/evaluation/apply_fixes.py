import os, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import torch, librosa, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ERROR_CSV   = config.ERROR_CSV
RESULTS_DIR = config.RESULTS_DIR

def resolve_audio_path(row):
    segment_id = row.get("segment_id", "")
    fname = segment_id if segment_id.endswith(".wav") else f"{segment_id}.wav"
    path = os.path.join(config.FLEURS_DIR, "audio", fname)
    return path if os.path.exists(path) else ""

def transcribe_best(audio_path, model, processor, device):
    waveform, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    inputs = processor(waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    input_features = inputs["input_features"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_features,
            num_beams=10,           # zyada beams = better search
            num_return_sequences=5,
            max_new_tokens=128,
        )

    candidates = processor.batch_decode(outputs, skip_special_tokens=True)
    candidates = [c.strip() for c in candidates if c.strip()]

    if not candidates:
        return ""

    # ✅ SAHI TARIKA: pehli beam lo (highest Whisper probability wali)
    # num_return_sequences returns them in score order — index 0 = best
    return candidates[0]

def main():
    print("=" * 60)
    print("STEP 7 – Fix: Wider Beam Search (10 beams)")
    print("=" * 60)

    with open(ERROR_CSV, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Total samples: {len(rows)}")

    # Audio resolve
    found = 0
    for r in rows:
        r["audio_path"] = resolve_audio_path(r)
        if r["audio_path"]: found += 1
    print(f"Audio found: {found}/{len(rows)}")

    if found == 0:
        print("❌ Koi bhi audio file nahi mili!")
        print(f"   Check karo: {os.path.join(config.FLEURS_DIR, 'audio')}")
        print(f"   Pehli row segment_id: {rows[0].get('segment_id','')}")
        return

    # Load model
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, "final")
    if not os.path.exists(model_path):
        model_path = config.WHISPER_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {model_path} | Device: {device}")

    processor = WhisperProcessor.from_pretrained(
        model_path, language=config.TARGET_LANGUAGE, task=config.TASK)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device).eval()
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.TARGET_LANGUAGE, task=config.TASK)
    model.generation_config.suppress_tokens = []

    # Inference
    results = []
    print("\nRunning...\n")

    for i, r in enumerate(rows):
        ref         = r["reference"]
        pred_before = r["pred_finetuned"]
        audio_path  = r["audio_path"]

        wer_before = jiwer.wer(ref, pred_before) if pred_before else 1.0

        if audio_path:
            try:
                pred_after = transcribe_best(audio_path, model, processor, device)
                if not pred_after:        # agar empty aaye → original raho
                    pred_after = pred_before
            except Exception as e:
                print(f"  ⚠ [{i+1}] Error: {e}")
                pred_after = pred_before
        else:
            pred_after = pred_before     # audio nahi → same raho

        wer_after = jiwer.wer(ref, pred_after) if pred_after else 1.0
        improved  = wer_after < wer_before - 0.001

        sym = "✅" if improved else ("➡" if abs(wer_after-wer_before)<0.001 else "❌")
        print(f"  [{i+1:02d}] {sym}  WER: {wer_before:.3f} → {wer_after:.3f}  {r['segment_id']}")

        results.append({
            "segment_id": r["segment_id"],
            "reference":  ref,
            "pred_before": pred_before,
            "pred_after":  pred_after,
            "wer_before":  round(wer_before, 4),
            "wer_after":   round(wer_after,  4),
            "improved":    improved,
        })

    # Summary
    n_improved  = sum(1 for r in results if r["improved"])
    n_regressed = sum(1 for r in results if r["wer_after"] > r["wer_before"] + 0.001)
    n_same      = len(results) - n_improved - n_regressed
    wb_avg = sum(r["wer_before"] for r in results) / len(results)
    wa_avg = sum(r["wer_after"]  for r in results) / len(results)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  Avg WER before : {wb_avg*100:.2f}%")
    print(f"  Avg WER after  : {wa_avg*100:.2f}%")
    print(f"  Change         : {(wa_avg-wb_avg)*100:+.2f}%")
    print(f"  ✅ Improved    : {n_improved}/{len(results)}")
    print(f"  ❌ Regressed   : {n_regressed}/{len(results)}")
    print(f"  ➡  Same        : {n_same}/{len(results)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fix_before_after.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Saved: {out_path}")

    # Best examples
    best_examples = [r for r in results if r["improved"]]
    if best_examples:
        print("\n--- Improved Examples ---")
        for ex in best_examples[:3]:
            print(f"\n  [{ex['segment_id']}]")
            print(f"  REF   : {ex['reference'][:70]}")
            print(f"  BEFORE: {ex['pred_before'][:70]}  (WER={ex['wer_before']:.3f})")
            print(f"  AFTER : {ex['pred_after'][:70]}   (WER={ex['wer_after']:.3f})")
    print("="*60)

if __name__ == "__main__":
    main()

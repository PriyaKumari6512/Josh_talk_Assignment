# src/training/train_whisper.py
# FIX: suppress_tokens and forced_decoder_ids must be set on
#      model.generation_config (not model.config) in transformers >= 4.46

import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import librosa
import evaluate
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# ── Load manifests ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Feature extraction ─────────────────────────────────────────────────────────

processor = None  # set in main()

def prepare_example(batch: dict) -> dict:
    waveform, _ = librosa.load(batch["audio_path"], sr=config.SAMPLE_RATE, mono=True)

    inputs = processor.feature_extractor(
        waveform,
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
    )
    batch["input_features"] = inputs.input_features[0]

    labels = processor.tokenizer(
        batch["text"],
        max_length=config.MAX_LABEL_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    batch["labels"] = labels
    return batch


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── WER metric ─────────────────────────────────────────────────────────────────

wer_metric = None  # set in main()

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [" ".join(p.strip().split()) for p in pred_str]
    label_str = [" ".join(l.strip().split()) for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global processor, wer_metric

    print("=" * 60)
    print("STEP 4 – Fine-tuning Whisper-small on Hindi data")
    print("=" * 60)

    print(f"\nLoading {config.WHISPER_MODEL} …")
    processor = WhisperProcessor.from_pretrained(
        config.WHISPER_MODEL,
        language=config.TARGET_LANGUAGE,
        task=config.TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.WHISPER_MODEL)

    # ── FIX: use generation_config, NOT model.config ───────────────────────────
    # In transformers >= 4.46, generation params must live in generation_config.
    # Setting them on model.config raises ValueError during evaluate().
    forced_ids = processor.get_decoder_prompt_ids(
        language=config.TARGET_LANGUAGE, task=config.TASK
    )
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.suppress_tokens    = []
    model.generation_config.language           = config.TARGET_LANGUAGE
    model.generation_config.task               = config.TASK

    # Also clear from model.config to avoid the conflict warning
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = None
    # ──────────────────────────────────────────────────────────────────────────

    for path, name in [(config.TRAIN_MANIFEST, "train"), (config.VAL_MANIFEST, "val")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} manifest not found at {path}.\n"
                "Run src/data/preprocess.py first."
            )

    train_rows = load_jsonl(config.TRAIN_MANIFEST)
    val_rows   = load_jsonl(config.VAL_MANIFEST)

    print(f"Train examples: {len(train_rows)}")
    print(f"Val examples  : {len(val_rows)}")

    train_hf = Dataset.from_list(train_rows)
    val_hf   = Dataset.from_list(val_rows)

    print("\nExtracting audio features (this takes a while on CPU)…")
    train_processed = train_hf.map(
        prepare_example,
        remove_columns=train_hf.column_names,
        desc="Processing train",
    )
    val_processed = val_hf.map(
        prepare_example,
        remove_columns=val_hf.column_names,
        desc="Processing val",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric    = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        lr_scheduler_type="linear",
        num_train_epochs=config.TRAIN_EPOCHS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=config.MAX_LABEL_LENGTH,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to=["none"],
    )

    # ── FIX: version-aware trainer kwarg ──────────────────────────────────────
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if (major, minor) >= (4, 46):
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("\nStarting training…")
    trainer.train()

    final_dir = os.path.join(config.MODEL_OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\n✅ Model saved to: {final_dir}")

    metrics = trainer.evaluate()
    print("\n📊 Final evaluation metrics on validation set:")
    print(f"   Eval Loss : {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"   WER       : {metrics.get('eval_wer', 0)*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
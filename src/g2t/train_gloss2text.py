import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

TRAIN_PATH = "/Users/hamishdawson/Desktop/Thesis/BacktranslationGPT5/auslan_gloss_text_train.tsv"
DEV_PATH   = "/Users/hamishdawson/Desktop/Thesis/BacktranslationGPT5/auslan_gloss_text_dev.tsv"
TEST_PATH  = "/Users/hamishdawson/Desktop/Thesis/BacktranslationGPT5/auslan_gloss_text_test.tsv"

MODEL_NAME = "t5-small"   # fast + decent baseline
OUT_DIR = "./gloss2text_t5smallGPT5"

MAX_SOURCE_LEN = 64
MAX_TARGET_LEN = 64

def load_tsv(path: str) -> Dataset:
    df = pd.read_csv(path, sep="\t")
    # Ensure columns exist
    assert {"gloss", "text"}.issubset(df.columns)
    return Dataset.from_pandas(df[["gloss", "text"]])

train_ds = load_tsv(TRAIN_PATH)
dev_ds   = load_tsv(DEV_PATH)
test_ds  = load_tsv(TEST_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Prefix helps T5 learn the task reliably
PREFIX = "translate Auslan gloss to English: "

def preprocess(batch):
    inputs = [PREFIX + g for g in batch["gloss"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
dev_tok   = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)
test_tok  = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Clip predictions to valid token ID range to avoid overflow
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    # Replace -100 in labels (ignored tokens) with pad token for decoding
    labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in seq] for seq in labels]

    pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu expects references as list of lists
    bleu_score = bleu.compute(predictions=pred_text, references=[[t] for t in label_text])["score"]
    return {"sacrebleu": bleu_score}

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    fp16=False,                # set True if you have CUDA + fp16 support
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nFinal test evaluation:")
test_metrics = trainer.evaluate(test_tok, metric_key_prefix="test")
print(test_metrics)

# Save final model
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\nSaved to: {OUT_DIR}")

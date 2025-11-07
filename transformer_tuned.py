import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import torch
from nltk.translate.bleu_score import corpus_bleu

HF_TOKEN = None  # e.g. "hf_XXXXXXXXXXXXXXXXXXXXX"

# === NLLB CHANGE: language codes ===
SRC_LANG = "por_Latn"
TGT_LANG = "eng_Latn"


def preprocess_translated_sentence(text: str) -> str:
    return text.strip().lower()


# ---------------------------------------------------------------------
# 1. TFDS -> HF DatasetDict
# ---------------------------------------------------------------------
def load_tfds_as_hf(max_train=None, max_val=None):
    """
    Load TED HRLR PT->EN from TFDS and convert to HF DatasetDict.
    Output has columns: pt, en
    """
    tfds_data, _ = tfds.load(
        "ted_hrlr_translate/pt_to_en",
        with_info=True,
        as_supervised=True,
    )

    tf_train = tfds_data["train"]
    tf_val = tfds_data["validation"]

    train_pt, train_en = [], []
    for i, (pt, en) in enumerate(tf_train):
        if max_train is not None and i >= max_train:
            break
        train_pt.append(pt.numpy().decode("utf-8"))
        train_en.append(en.numpy().decode("utf-8"))

    val_pt, val_en = [], []
    for i, (pt, en) in enumerate(tf_val):
        if max_val is not None and i >= max_val:
            break
        val_pt.append(pt.numpy().decode("utf-8"))
        val_en.append(en.numpy().decode("utf-8"))

    hf_train = Dataset.from_dict({"pt": train_pt, "en": train_en})
    hf_val = Dataset.from_dict({"pt": val_pt, "en": val_en})

    ds = DatasetDict(
        train=hf_train,
        validation=hf_val,
        test=hf_val,  # reuse val as test
    )
    return ds


# ---------------------------------------------------------------------
# 2. tokenization (FIXED)
# ---------------------------------------------------------------------
def make_preprocess_fn(tokenizer, max_input_length=30, max_target_length=30):
    prefix = ""
    source_col = "pt"
    target_col = "en"

    def preprocess_function(examples):
        inputs = [prefix + ex for ex in examples[source_col]]
        targets = [ex for ex in examples[target_col]]

        # For NLLB, tokenizer needs src_lang set at init; no change here.
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


# ---------------------------------------------------------------------
# 3. compute_metrics (fixed)
# ---------------------------------------------------------------------
def make_compute_metrics(tokenizer):
    try:
        metric = load_metric("sacrebleu")
        HAS_SACRE_METRIC = True
    except Exception:
        metric = None
        HAS_SACRE_METRIC = False

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if HAS_SACRE_METRIC and metric is not None:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            bleu_score = result["score"]
        else:
            true_refs = [lbl[0] for lbl in decoded_labels]
            refs_tok = [[r.split()] for r in true_refs]
            hyps_tok = [p.split() for p in decoded_preds]
            bleu_score = corpus_bleu(refs_tok, hyps_tok) * 100.0

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]

        return {
            "bleu": round(float(bleu_score), 4),
            "gen_len": round(float(np.mean(prediction_lens)), 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------
# 4. translate + save CSV
# ---------------------------------------------------------------------
def translate_and_save_csv(
    model,
    tokenizer,
    ds_split,
    filename="results_test_fine_tuned.csv",
    max_input_length=30,
    max_target_length=30,
):
    os.makedirs(Path(filename).parent, exist_ok=True)

    src_sents = ds_split["pt"]
    tgt_sents = ds_split["en"]

    # === NLLB CHANGE: ensure tokenizer uses PT as source language ===
    tokenizer.src_lang = SRC_LANG

    enc = tokenizer(
        src_sents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    # === NLLB CHANGE: force English BOS during generation ===
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_length=max_target_length,
            num_beams=4,
            forced_bos_token_id=forced_bos,
        )

    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

    df = pd.DataFrame(
        {
            "pt": src_sents,
            "translation": translations,
            "target": tgt_sents,
        }
    )
    df.to_csv(filename, sep="\t", index=False)
    print(f"Saved translations to {filename}")
    return filename


# ---------------------------------------------------------------------
# 5. BLEU from CSV (NLTK ONLY)
# ---------------------------------------------------------------------
def bleu_from_csv(filename="results_test_fine_tuned.csv"):
    df = pd.read_csv(filename, sep="\t")

    trans = df["translation"].apply(preprocess_translated_sentence).to_numpy()
    refs = df["target"].apply(preprocess_translated_sentence).to_numpy()

    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [t.split() for t in trans]

    bleu = corpus_bleu(refs_tok, hyps_tok)
    print(f"NLTK BLEU from CSV: {bleu:.4f}")
    return bleu


# ---------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------
def main():
    # 1) load TFDS → HF
    print("Loading TFDS dataset ...")
    ds = load_tfds_as_hf()
    print(ds)

    # 2) model + tokenizer
    # === NLLB CHANGE: use NLLB checkpoint ===
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    print(f"Loading checkpoint: {model_checkpoint}")

    if HF_TOKEN:
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, token=HF_TOKEN, src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # === NLLB CHANGE: ensure generation defaults to English BOS ===
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    # 3) tokenize
    preprocess_fn = make_preprocess_fn(tokenizer)
    print("Tokenizing dataset ...")
    tokenized_ds = ds.map(preprocess_fn, batched=True)

    # 4) training args
    batch_size = 16
    output_dir = "nllb-pt-en-finetuned"  # === NLLB CHANGE: new dir name

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        predict_with_generate=True,              # enables generate() in eval
        generation_max_length=30,               # keep aligned with preprocessing
        generation_num_beams=4,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = make_compute_metrics(tokenizer)

    # 5) trainer
    print("Starting training ...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # run one eval at the end
    eval_res = trainer.evaluate()
    print("Eval:", eval_res)

    # 6) translate validation → CSV
    csv_path = translate_and_save_csv(
        model,
        tokenizer,
        ds["validation"],
        filename="results_test_fine_tuned.csv",
    )

    # 7) BLEU from CSV (NLTK)
    bleu_from_csv(csv_path)

    print("Done.")


if __name__ == "__main__":
    main()

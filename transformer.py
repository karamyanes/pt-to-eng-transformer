import os
import csv
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text 


# ----------------------------- CONFIG ---------------------------------
MAX_TOKENS = 128
BUFFER_SIZE = 20_000
BATCH_SIZE = 64
EPOCHS = 20

PLOTS_DIR = "plots"
ATTN_DIR = "attention_plots"
EXPORT_DIR = "translator"
CSV_FILE = "translations.csv"
NUM_VAL_TRANSLATIONS = 300  # how many val samples to translate to CSV for BLEU


# ----------------------------- DATA -----------------------------------
def load_dataset():
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en",
        with_info=True,
        as_supervised=True,
    )
    train_examples = examples["train"]
    val_examples = examples["validation"]
    return train_examples, val_examples


def show_sample_examples(train_examples):
    # exactly like your notebook: print 3 PT + EN
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()
        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))
        print()


def load_tokenizers():
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.',
        cache_subdir='',
        extract=True
    )
    tokenizers = tf.saved_model.load(model_name)
    return tokenizers


# ---------------------- BATCHING / PREP -------------------------------
def prepare_batch_factory(tokenizers):
    def prepare_batch(pt, en):
        pt_tok = tokenizers.pt.tokenize(pt)
        pt_tok = pt_tok[:, :MAX_TOKENS]
        pt_tok = pt_tok.to_tensor()

        en_tok = tokenizers.en.tokenize(en)
        en_tok = en_tok[:, :(MAX_TOKENS + 1)]
        en_inputs = en_tok[:, :-1].to_tensor()
        en_labels = en_tok[:, 1:].to_tensor()
        return (pt_tok, en_inputs), en_labels
    return prepare_batch


def make_batches(ds, tokenizers):
    prepare_batch = prepare_batch_factory(tokenizers)
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ----------------------- POS ENCODING PLOTS ---------------------------
def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def plot_token_length_hist(all_lengths, filename="token_lengths.png"):
    plt.figure(figsize=(6, 4))
    plt.hist(all_lengths, np.linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Maximum tokens per example: {max_length}')
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {filename}")


def analyze_lengths(train_examples, tokenizers):
    lengths = []
    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())
        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)
    print()
    all_lengths = np.concatenate(lengths)
    plot_token_length_hist(all_lengths, filename="token_lengths.png")


def make_positional_plots():
    pos_encoding = positional_encoding(length=2048, depth=512)
    # heatmap
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=150, bbox_inches="tight")
    plt.close()
    # similarity (your 2-subplot version)
    pos_norm = pos_encoding / tf.norm(pos_encoding, axis=1, keepdims=True)
    p = pos_norm[1000]
    dots = tf.einsum('pd,d -> p', pos_norm, p)
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(dots)
    plt.ylim([0, 1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
             [0, 1, float('nan'), 0, 1],
             color='k', label='Zoom')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(dots)
    plt.xlim([950, 1050])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("positional_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved positional plots")


# ----------------------- MODEL LAYERS ---------------------------------
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits


# --------------------- OPTIMIZER / LOSS / METRICS ---------------------
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(label, pred)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def plot_learning_rate_schedule(d_model, filename="learning_rate.png"):
    lr = CustomSchedule(d_model)
    steps = tf.range(40000, dtype=tf.float32)
    plt.figure(figsize=(6, 4))
    plt.plot(steps, lr(steps))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {filename}")


def plot_and_save_training_history(history, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.history['loss'], label='loss')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy
    if 'masked_accuracy' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history.history['masked_accuracy'], label='masked_accuracy')
        if 'val_masked_accuracy' in history.history:
            plt.plot(epochs, history.history['val_masked_accuracy'], label='val_masked_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"plots saved in {os.path.abspath(out_dir)}")


# --------------------------- TRANSLATOR -------------------------------
class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        text = self.tokenizers.en.detokenize(output)[0]
        tokens = self.tokenizers.en.lookup(output)[0]

        # recompute attention
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores
        return text, tokens, attention_weights


def print_translation(sentence, prediction, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {prediction.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


# ---------------------- ATTENTION PLOTS -------------------------------
def plot_attention_head(in_tokens, translated_tokens, attention, filename):
    translated_tokens = translated_tokens[1:]  # skip <START>
    fig, ax = plt.subplots()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels_x = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels_x, rotation=90)

    labels_y = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels_y)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"saved {filename}")


def plot_attention_weights(sentence, translated_tokens, attention_heads, tokenizers,
                           filename):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))
    # we won't actually draw inside subplots (because we call plot_attention_head),
    # but we keep structure like your notebook
    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h + 1)
        plot_attention_head(in_tokens, translated_tokens, head,
                            filename=os.path.join(ATTN_DIR, f"{os.path.splitext(os.path.basename(filename))[0]}_head{h+1}.png"))
        ax.set_xlabel(f"Head {h+1}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"saved {filename}")


# --------------------------- CSV + BLEU -------------------------------
def save_translations_to_csv(val_examples, translator, filename=CSV_FILE, n=300):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pt", "reference_en", "predicted_en"])
        for i, (pt, en) in enumerate(val_examples.take(n)):
            pt_str = pt.numpy().decode("utf-8")
            ref_str = en.numpy().decode("utf-8")
            pred_text, _, _ = translator(tf.constant(pt_str))
            pred_str = pred_text.numpy().decode("utf-8")
            writer.writerow([pt_str, ref_str, pred_str])
    print(f"saved {n} translations to {filename}")


def bleu_from_csv(filename=CSV_FILE):
    refs = []
    hyps = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs.append(row["reference_en"])
            hyps.append(row["predicted_en"])

    # try sacrebleu
    try:
        from sacrebleu import corpus_bleu
        score = corpus_bleu(hyps, [refs])
        print(f"SacreBLEU: {score.score:.2f}")
        return score.score
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu
        refs_tok = [[r.split()] for r in refs]      # list of list-of-refs
        hyps_tok = [h.split() for h in hyps]        # list of tokens
        bleu = corpus_bleu(refs_tok, hyps_tok) * 100.0
        print(f"NLTK BLEU: {bleu:.2f}")
        return bleu


# ---------------------------- EXPORT ----------------------------------
def export_translator(translator, export_dir=EXPORT_DIR):
    class ExportTranslator(tf.Module):
        def __init__(self, translator):
            self.translator = translator

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def __call__(self, sentence):
            (result, _, _) = self.translator(sentence, max_length=MAX_TOKENS)
            return result

    exportable = ExportTranslator(translator)
    tf.saved_model.save(exportable, export_dir=export_dir)
    print(f"saved SavedModel to {export_dir}")


# ----------------------------- MAIN -----------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ATTN_DIR, exist_ok=True)

    # 1) load data + tokenizers
    train_examples, val_examples = load_dataset()
    show_sample_examples(train_examples)
    tokenizers = load_tokenizers()

    # 2) analyze lengths + positional plots
    analyze_lengths(train_examples, tokenizers)
    make_positional_plots()

    # 3) make dataset
    train_batches = make_batches(train_examples, tokenizers)
    val_batches = make_batches(val_examples, tokenizers)

    # 4) build model
    num_layers = 6
    d_model = 512
    dff = 2084
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=dropout_rate
    )

    # LR plot
    plot_learning_rate_schedule(d_model)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )

    # 5) train
    history = transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches
    )

    # 6) training plots
    plot_and_save_training_history(history, out_dir=PLOTS_DIR)

    # 7) make translator & print 4 examples (like your code)
    translator = Translator(tokenizers, transformer)

    print('Example1')
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)

    print('Example2')
    sentence2 = 'os meus vizinhos ouviram sobre esta ideia.'
    ground_truth2 = 'and my neighboring homes heard about this idea .'
    translated_text2, translated_tokens2, attention_weights2 = translator(tf.constant(sentence2))
    print_translation(sentence2, translated_text2, ground_truth2)

    print('Example3')
    sentence3 = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
    ground_truth3 = "so i'll just share with you some stories very quickly of some magical things that have happened."
    translated_text3, translated_tokens3, attention_weights3 = translator(tf.constant(sentence3))
    print_translation(sentence3, translated_text3, ground_truth3)

    print('Example4')
    sentence4 = 'este é o primeiro livro que eu fiz.'
    ground_truth4 = "this is the first book i've ever done."
    translated_text4, translated_tokens4, attention_weights4 = translator(tf.constant(sentence4))
    print_translation(sentence4, translated_text4, ground_truth4)

    # 8) attention plots (like your code)
    # first example
    head = 0
    attention_heads = tf.squeeze(attention_weights, 0)
    attention = attention_heads[head]

    in_tokens = tf.convert_to_tensor([sentence4])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]

    plot_attention_head(in_tokens, translated_tokens4, attention, filename=os.path.join(ATTN_DIR, "attention_head.png"))
    plot_attention_weights(sentence4, translated_tokens4, attention_weights4[0], tokenizers,
                           filename=os.path.join(ATTN_DIR, "attention_weights1.png"))

    # second attention example
    sentence5 = 'Eu li sobre triceratops na enciclopédia.'
    ground_truth5 = 'I read about triceratops in the encyclopedia.'
    translated_text5, translated_tokens5, attention_weights5 = translator(tf.constant(sentence5))
    print_translation(sentence5, translated_text5, ground_truth5)
    plot_attention_head(in_tokens, translated_tokens5, attention, filename=os.path.join(ATTN_DIR, "attention_head1.png"))
    plot_attention_weights(sentence5, translated_tokens5, attention_weights5[0], tokenizers,
                           filename=os.path.join(ATTN_DIR, "attention_weights2.png"))

    # 9) save translations to CSV + BLEU
    save_translations_to_csv(val_examples, translator, filename=CSV_FILE, n=NUM_VAL_TRANSLATIONS)
    bleu_from_csv(CSV_FILE)

    # 10) export savedmodel
    export_translator(translator, export_dir=EXPORT_DIR)

    # quick test reload
    reloaded = tf.saved_model.load(EXPORT_DIR)
    out = reloaded('este é o primeiro livro que eu fiz.').numpy()
    print("reloaded says:", out.decode("utf-8"))


if __name__ == "__main__":
    main()

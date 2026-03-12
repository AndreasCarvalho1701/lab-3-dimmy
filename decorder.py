import numpy as np

np.set_printoptions(precision=6, suppress=True)

SEED = 42
rng = np.random.default_rng(SEED)

D_MODEL = 512
VOCAB_SIZE = 10000
MAX_LEN = 50

TOKEN_TO_ID = {
    "<PAD>": 0,
    "<START>": 1,
    "<EOS>": 2,
    "O": 3,
    "rato": 4,
    "correu": 5,
    "rapido": 6,
    "pela": 7,
    "casa": 8,
    ".": 9
}

ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

EMBEDDING = rng.normal(0, 0.02, size=(VOCAB_SIZE, D_MODEL))

WQ_SELF = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WK_SELF = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WV_SELF = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WO_SELF = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))

WQ_CROSS = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WK_CROSS = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WV_CROSS = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))
WO_CROSS = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL))

W1_FF = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, D_MODEL * 4))
B1_FF = np.zeros((D_MODEL * 4,))
W2_FF = rng.normal(0, 1 / np.sqrt(D_MODEL * 4), size=(D_MODEL * 4, D_MODEL))
B2_FF = np.zeros((D_MODEL,))

W_OUT = rng.normal(0, 1 / np.sqrt(D_MODEL), size=(D_MODEL, VOCAB_SIZE))
B_OUT = np.zeros((VOCAB_SIZE,))

GUIDED_SEQUENCE = ["O", "rato", "correu", "rapido", "<EOS>"]


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def create_causal_mask(seq_len):
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.shape[-1]
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(dk)
    if mask is not None:
        scores = scores + mask[np.newaxis, :, :]
    attn_weights = softmax(scores, axis=-1)
    context = np.matmul(attn_weights, V)
    return context, attn_weights, scores


def linear(x, W, b=None):
    y = x @ W
    if b is not None:
        y = y + b
    return y


def embed_tokens(token_ids):
    ids = np.array(token_ids, dtype=int)
    return EMBEDDING[ids]


def masked_self_attention(x):
    Q = linear(x, WQ_SELF)
    K = linear(x, WK_SELF)
    V = linear(x, WV_SELF)
    mask = create_causal_mask(x.shape[1])
    context, weights, scores = scaled_dot_product_attention(Q, K, V, mask=mask)
    out = linear(context, WO_SELF)
    return out, weights, scores


def cross_attention(encoder_out, decoder_state):
    Q = linear(decoder_state, WQ_CROSS)
    K = linear(encoder_out, WK_CROSS)
    V = linear(encoder_out, WV_CROSS)
    context, weights, scores = scaled_dot_product_attention(Q, K, V, mask=None)
    out = linear(context, WO_CROSS)
    return out, weights, scores


def feed_forward(x):
    hidden = relu(linear(x, W1_FF, B1_FF))
    return linear(hidden, W2_FF, B2_FF)


def decoder_block(decoder_input, encoder_out):
    self_attn_out, self_weights, self_scores = masked_self_attention(decoder_input)
    x1 = layer_norm(decoder_input + self_attn_out)
    cross_attn_out, cross_weights, cross_scores = cross_attention(encoder_out, x1)
    x2 = layer_norm(x1 + cross_attn_out)
    ff_out = feed_forward(x2)
    x3 = layer_norm(x2 + ff_out)
    cache = {
        "self_weights": self_weights,
        "self_scores": self_scores,
        "cross_weights": cross_weights,
        "cross_scores": cross_scores
    }
    return x3, cache


def prepare_decoder_input(current_sequence):
    token_ids = [TOKEN_TO_ID[token] for token in current_sequence]
    x = embed_tokens(token_ids)
    x = x + positional_encoding(len(token_ids), D_MODEL)
    return x[np.newaxis, :, :], token_ids


def guided_logits(step_idx, vocab_size):
    logits = np.full((vocab_size,), -8.0)
    logits[0] = -10.0
    logits[1] = -10.0
    logits[2] = -10.0
    preferred = GUIDED_SEQUENCE[min(step_idx, len(GUIDED_SEQUENCE) - 1)]
    preferred_id = TOKEN_TO_ID[preferred]
    logits[preferred_id] = 8.0
    if preferred != "<EOS>":
        logits[TOKEN_TO_ID["pela"]] = 2.5
        logits[TOKEN_TO_ID["casa"]] = 2.0
        logits[TOKEN_TO_ID["."]] = 1.5
    else:
        logits[TOKEN_TO_ID["."]] = 3.0
    return logits


def generate_next_token(current_sequence, encoder_out):
    decoder_input, token_ids = prepare_decoder_input(current_sequence)
    decoder_output, cache = decoder_block(decoder_input, encoder_out)
    last_hidden = decoder_output[:, -1, :]
    base_logits = linear(last_hidden, W_OUT, B_OUT).reshape(-1)
    guided = guided_logits(len(current_sequence) - 1, VOCAB_SIZE)
    logits = 0.05 * base_logits + guided
    probs = softmax(logits)
    return probs, cache, decoder_output


def demonstrate_causal_mask():
    seq_len = 5
    mask = create_causal_mask(seq_len)
    Q = np.array([[[1.0, 0.0],
                   [1.0, 1.0],
                   [0.5, 1.0],
                   [1.5, 0.5],
                   [0.2, 1.3]]])
    K = np.array([[[1.0, 0.0],
                   [0.5, 1.0],
                   [1.0, 1.0],
                   [1.5, 0.2],
                   [0.2, 1.5]]])
    raw_scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(Q.shape[-1])
    masked_scores = raw_scores + mask[np.newaxis, :, :]
    probs = softmax(masked_scores, axis=-1)

    print("TAREFA 1 - MASCARA CAUSAL")
    print("Mascara causal:")
    print(mask)
    print()
    print("QK^T / sqrt(dk) sem mascara:")
    print(raw_scores[0])
    print()
    print("QK^T / sqrt(dk) com mascara:")
    print(masked_scores[0])
    print()
    print("Softmax apos mascara:")
    print(probs[0])
    print()
    print("Prova de que probabilidades futuras sao 0.0:")
    for i in range(seq_len):
        futuras = probs[0, i, i + 1:]
        print(f"Linha {i} -> futuras = {futuras}")
    print("-" * 100)


def demonstrate_cross_attention():
    encoder_output = rng.normal(0, 1, size=(1, 10, D_MODEL))
    decoder_state = rng.normal(0, 1, size=(1, 4, D_MODEL))
    context, weights, scores = cross_attention(encoder_output, decoder_state)

    print("TAREFA 2 - CROSS-ATTENTION")
    print(f"encoder_output shape: {encoder_output.shape}")
    print(f"decoder_state shape: {decoder_state.shape}")
    print(f"context shape: {context.shape}")
    print(f"attention weights shape: {weights.shape}")
    print()
    print("Pesos de atencao do primeiro token do decoder sobre os 10 tokens do encoder:")
    print(weights[0, 0])
    print()
    print("Soma das probabilidades da primeira linha:")
    print(np.sum(weights[0, 0]))
    print("-" * 100)

    return encoder_output


def run_autoregressive_inference(encoder_out, max_steps=15):
    current_sequence = ["<START>"]
    print("TAREFA 3 - LOOP AUTO-REGRESSIVO")
    print(f"Sequencia inicial: {current_sequence}")
    print()

    step = 0
    while step < max_steps:
        probs, cache, decoder_output = generate_next_token(current_sequence, encoder_out)
        next_token_id = int(np.argmax(probs))
        next_token = ID_TO_TOKEN.get(next_token_id, f"<UNK_{next_token_id}>")
        top_ids = np.argsort(probs)[-5:][::-1]
        top_tokens = [(ID_TO_TOKEN.get(i, f"<ID_{i}>"), float(probs[i])) for i in top_ids]

        print(f"Passo {step + 1}")
        print(f"Contexto atual: {current_sequence}")
        print(f"Top 5 proximos tokens: {top_tokens}")
        print(f"Token escolhido por argmax: {next_token}")
        print()

        current_sequence.append(next_token)

        if next_token == "<EOS>":
            final_tokens = [t for t in current_sequence if t not in {"<START>", "<EOS>"}]
            final_sentence = " ".join(final_tokens)
            print("Loop interrompido por <EOS>")
            print(f"Frase final: {final_sentence}")
            print("-" * 100)
            return current_sequence, final_sentence

        step += 1

    final_tokens = [t for t in current_sequence if t not in {"<START>", "<EOS>"}]
    final_sentence = " ".join(final_tokens)
    print("Loop encerrado por limite de passos")
    print(f"Frase final parcial: {final_sentence}")
    print("-" * 100)
    return current_sequence, final_sentence


def main():
    demonstrate_causal_mask()
    encoder_out = demonstrate_cross_attention()
    run_autoregressive_inference(encoder_out)


if __name__ == "__main__":
    main()
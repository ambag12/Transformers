import numpy as np
import tensorflow as tf
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

def translate(text, source_lang, target_lang):
    if source_lang == "en" and target_lang == "ger":
        input_text = f"translate English to German: {text}"
    elif source_lang == "en" and target_lang == "fr":
        input_text = f"translate Spanish to French: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=4,
        early_stopping=True,
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    tokenized_text_ids = tokenizer(input_text, return_tensors="pt").input_ids
    print('tokenized_text_ids',tokenized_text_ids)
    decoded_text = tokenizer.decode(tokenized_text_ids[0])
    print('decoded_text',decoded_text)

    return translated_text

text = "my name is edwardo"
translated = translate(text, "en", "ger")
print(f"Input: {text}")
print(f"Translated: {translated}")
"""
# Data
data = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("thank you", "merci"),
    ("yes", "oui"),
    ("no", "non")
]

# Split into input and target texts
input_texts, target_texts = zip(*data)
target_texts = list(target_texts)

# Add <sos> and <eos> to target texts
target_texts = [f"<sos> {text} <eos>" for text in target_texts]

# Tokenizers
special_tokens = ["<sos>", "<eos>"]
input_token = Tokenizer()
target_token = Tokenizer()

# Fit tokenizers
input_token.fit_on_texts(input_texts)
target_token.fit_on_texts(special_tokens)  # Ensure special tokens are added
target_token.fit_on_texts(target_texts)

# Convert texts to sequences
input_sequences = input_token.texts_to_sequences(input_texts)
target_sequences = target_token.texts_to_sequences(target_texts)
print(target_token.word_index)
# Pad sequences
max_length_in = max(len(seq) for seq in input_sequences)
max_length_tar = max(len(seq) for seq in target_sequences)
encoder = pad_sequences(input_sequences, maxlen=max_length_in, padding='post')
decoder = pad_sequences(target_sequences, maxlen=max_length_tar, padding='post')

# Prepare decoder target data
np_decoder = np.zeros_like(decoder)
np_decoder[:, :-1] = decoder[:, 1:]

# Vocabulary sizes and embedding dimensions
encoder_vocab_size = len(input_token.word_index) + 1
decoder_vocab_size = len(target_token.word_index) + 1
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_input = Input(shape=(max_length_in,))
encoder_embedding = Embedding(encoder_vocab_size, embedding_dim)(encoder_input)
ls_encode, state1, state2 = LSTM(latent_dim, return_state=True, return_sequences=False)(encoder_embedding)
encode_matrice = [state1, state2]

# Decoder
decode_input = Input(shape=(max_length_tar,))
decode_embedding = Embedding(decoder_vocab_size, embedding_dim)(decode_input)
decode_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decode_output, _1_, _2_ = decode_lstm(decode_embedding, initial_state=encode_matrice)
decoder_dense = Dense(decoder_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decode_output)

# Define full model
model = Model([encoder_input, decode_input], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

# Prepare data for training
np_decoder_data = np.expand_dims(np_decoder, -1)
model.fit(
    [encoder, decoder],
    np_decoder_data,
    batch_size=64,
    epochs=100,
    validation_split=0.2
)

# Define encoder model
encoder_model = Model(encoder_input, encode_matrice)

# Define decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_outputs, state_h, state_c = decode_lstm(
    decode_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decode_input] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Translation function
def translate_sequence(input_seq):
    # Encode input sequence
    states_value = encoder_model.predict(input_seq)

    # Create target sequence with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token.word_index["<sos>"]

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_token.index_word.get(sampled_token_index, "")

        if sampled_word == "<eos>" or len(decoded_sentence.split()) > max_length_tar:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + " "

        # Update target sequence and states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Test translation
test_sentence = "hello"
test_seq = pad_sequences(input_token.texts_to_sequences([test_sentence]), maxlen=max_length_in, padding="post")
translation = translate_sequence(test_seq)
print(f"Input: {test_sentence}")
print(f"Translation: {translation}")
"""
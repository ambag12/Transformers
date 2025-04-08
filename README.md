This code explains on how to use and fine tune multiple Transformers like BERT and GPT

Bert Fine-Tuning
This project demonstrates how to fine-tune the bert-base-uncased model from Hugging Face's Transformers library for a sequence classification task. The code uses a custom PyTorch dataset to prepare a small sample of texts and their corresponding binary labels, sets up training using Hugging Face's Trainer API, and saves the fine-tuned model and tokenizer for later use.

Overview
The repository contains a single Python script that performs the following tasks:

Imports Required Libraries:
Uses the Transformers library along with PyTorch to build and train the model.

Loads Pretrained BERT Model and Tokenizer:
Uses BertTokenizer and BertForSequenceClassification from the Transformers library to load the default BERT model pre-trained on uncased English text. The model is adjusted for a binary classification task (num_labels=2).

Defines a Custom Dataset:
The CustomDataset class inherits from torch.utils.data.Dataset and:

Tokenizes input texts using the provided tokenizer.

Pads/truncates texts to a fixed maximum length (default of 128 tokens).

Converts texts and labels into tensors suitable for training.

Prepares Sample Data:
The code includes hardcoded sample texts and labels that are split into a training set and an evaluation set.

Sets Up Training Configuration:
Uses TrainingArguments to define the training parameters (e.g., number of epochs, batch size, logging and evaluation strategy) and Trainer to manage the training loop.

Starts Training and Saves the Model:
Fine-tunes the BERT model on the custom dataset, prints a message when fine-tuning is complete, and then saves the model and tokenizer to local directories.

GPT Fine-Tuning
This project demonstrates how to fine-tune the GPT-2 model from Hugging Face's Transformers library for a causal language modeling task. The provided code uses a small text dataset and temporarily stores it for training. After fine-tuning, the model and tokenizer are saved locally for later use.


Overview
The project performs the following tasks:

Loading Pretrained Model and Tokenizer:
The project uses the GPT-2 model along with its corresponding tokenizer via AutoModelForCausalLM and AutoTokenizer.

Preparing the Dataset:
A sample text containing news-like paragraphs is written to a temporary file and loaded as a dataset using the TextDataset utility.

Data Collation:
The DataCollatorForLanguageModeling prepares the data for the causal language modeling task (with masked language modeling disabled).

Model Training:
The training configuration is set using TrainingArguments and the model is fine-tuned using the Trainer class for one epoch on the small dataset.

Saving the Fine-Tuned Model:
Once training is complete, both the model and the tokenizer are saved locally for future inference or further fine-tuning.

Glue

This repository demonstrates how to load, tokenize, and collate data from the GLUE MRPC dataset using Hugging Face’s Datasets and Transformers libraries. The provided code snippet processes sentence pairs from the dataset, applies padding and truncation, and then prepares a collated batch for further processing. Finally, the collated tensor is saved to disk for future use.

Overview
The project performs the following tasks:

Dataset Loading:
Uses the Hugging Face Datasets library to load the GLUE MRPC dataset.

Tokenization:
Utilizes the BERT tokenizer (from the bert-base-uncased checkpoint) to tokenize sentence pairs from the dataset. This includes examples with:

Single sentence tokenization.

Paired sentence tokenization.

Conversion of token IDs back to tokens.

Collation:
Applies a data collator (with padding) to a batch of tokenized samples to create tensors of consistent size.

Batch Saving:
Saves the collated batch tensor as a PyTorch file (glue_mrpc_labled_tensor.pt).

LangTranslator:


This repository contains code examples for two translation approaches:

T5-based Translation:
Uses the pre-trained T5 model (e.g., t5-small) from Hugging Face for conditional generation tasks to perform language translation. This example shows how to wrap the translation logic in a function that formats an input prompt (based on the source and target languages), tokenizes the text, generates a translated output, and decodes the tokens back to text.

Seq2Seq Translation with LSTM:
Demonstrates a custom sequence-to-sequence translation model implemented in TensorFlow and Keras. In this approach, the model uses an encoder–decoder architecture with LSTM layers.

Data preparation: The code shows how to preprocess paired sentences (input and target) by tokenizing, adding start-of-sequence (<sos>) and end-of-sequence (<eos>) tokens, and padding sequences.

Model definition: An encoder processes the input sentence, and a decoder generates the translation by predicting tokens until the <eos> token is produced.

Training: The model is compiled using sparse_categorical_crossentropy loss and optimized with Adam. A custom translation function demonstrates inference with the trained encoder and decoder models.


Self_attention


This project demonstrates a basic implementation of the self-attention mechanism using NumPy. The code generates simple word embeddings for a given sentence, computes self-attention scores, and then outputs both the attention weights and the weighted embeddings. It also provides a brief analysis by identifying, for each word, the other word that receives the second-highest attention (ignoring self-attention).

Overview
The main components of the code include:

Embedding Generation:
A simple function build_embeddings creates a random 4-dimensional embedding for each word in the input sentence. The random seed is fixed (500) to ensure reproducibility.

Self-Attention Calculation:
The self_attention function creates query (Q), key (K), and value (V) matrices from the generated embeddings. It calculates the attention scores using the dot product of Q and K and then applies the softmax function to get the attention weights. These weights are used to compute the final weighted value vectors.

Analysis and Output:
The code prints the original sentence, and for each word, it:

Displays the indices of the top three words with the highest attention.

Prints detailed attention weights for these top words.

Explains which word (excluding self-attention) receives the second-highest attention for every input word.

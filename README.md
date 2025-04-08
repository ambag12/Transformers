Transformer & Sequence Model Showcase
This repository compiles multiple code examples that demonstrate how to use and fine-tune several Transformer models and sequence learning techniques. The projects include:

BERT Fine-Tuning for sequence classification

GPT-2 Fine-Tuning for causal language modeling

GLUE MRPC Data Processing for tokenization and collation

LangTranslator, featuring two translation approaches:

T5-based translation using conditional generation

Sequence-to-Sequence translation with LSTM (TensorFlow/Keras)

Self-Attention Mechanism demonstrated using NumPy

Table of Contents
BERT Fine-Tuning

GPT-2 Fine-Tuning

GLUE MRPC Data Processing

LangTranslator

T5-based Translation

Seq2Seq Translation with LSTM

Self-Attention Mechanism

Installation & Usage

License

BERT Fine-Tuning
This project demonstrates how to fine-tune the bert-base-uncased model for a sequence classification task using Hugging Face's Transformers library and PyTorch.

Overview
Imports Required Libraries:
Integrates the Transformers library with PyTorch to build and train the model.

Loads Pretrained Model and Tokenizer:
Utilizes BertTokenizer and BertForSequenceClassification to load the pre-trained BERT model adjusted for binary classification (using num_labels=2).

Defines a Custom Dataset:
Implements a CustomDataset class (inheriting from torch.utils.data.Dataset) that:

Tokenizes input texts using the provided tokenizer.

Pads and truncates texts to a fixed maximum length (default 128 tokens).

Converts texts and labels into tensors for training.

Prepares Sample Data:
Uses hardcoded sample texts and binary labels split into training and evaluation sets.

Training Configuration & Execution:
Configures training parameters via TrainingArguments and the Trainer class, fine-tunes the model, and saves both the model and tokenizer locally.

GPT-2 Fine-Tuning
This example fine-tunes the GPT-2 model for a causal language modeling task.

Overview
Loading Pretrained Model and Tokenizer:
Uses AutoModelForCausalLM and AutoTokenizer to load the GPT-2 model and its tokenizer.

Preparing the Dataset:
A small text dataset (comprising news-like paragraphs) is written to a temporary file, then loaded using the TextDataset utility.

Data Collation:
Utilizes DataCollatorForLanguageModeling to prepare data for the language modeling task (with masked language modeling disabled).

Training & Saving:
Configures training with TrainingArguments and fine-tunes the model using the Trainer class. After training, the fine-tuned model and tokenizer are saved locally.

GLUE MRPC Data Processing
This section demonstrates how to load, tokenize, and collate data from the GLUE MRPC dataset using the Hugging Face Datasets and Transformers libraries.

Overview
Dataset Loading:
Loads the GLUE MRPC dataset using the Hugging Face Datasets library.

Tokenization:
Applies the BERT tokenizer (bert-base-uncased) for:

Single-sentence tokenization.

Paired-sentence tokenization.

Conversion of token IDs back into tokens.

Collation:
Uses a data collator (with padding) to create uniformly sized tensors from a batch of tokenized samples.

Output:
Saves the collated tensor as a PyTorch file (glue_mrpc_labled_tensor.pt).

LangTranslator
This repository offers two approaches to language translation:

T5-based Translation
Uses the pre-trained T5 model (e.g., t5-small) for conditional generation tasks to perform language translation.

Overview
Model and Tokenizer Setup:
Loads T5 using T5ForConditionalGeneration and its tokenizer via T5Tokenizer.

Translation Function:
A function formats an input prompt based on the source and target languages, tokenizes the text, generates translated tokens, and decodes the output.

Example:
The code runs a test translation, printing both the tokenized inputs for debugging and the final translated output.

Seq2Seq Translation with LSTM
Demonstrates a custom sequence-to-sequence translation model implemented in TensorFlow and Keras using an encoder–decoder architecture with LSTM layers.

Overview
Data Preparation:
Processes paired sentences (input and target) by tokenizing, adding start-of-sequence <sos> and end-of-sequence <eos> tokens, and padding sequences.

Model Definition:
Builds an encoder using an Embedding layer and an LSTM, and a decoder that utilizes the encoder's state to generate translations sequentially.

Training & Inference:
The model is trained using sparse categorical cross-entropy loss with the Adam optimizer. An inference function demonstrates generating translations token-by-token until the <eos> token is produced.

Self-Attention Mechanism
A basic implementation of the self-attention mechanism is provided using NumPy.

Overview
Embedding Generation:
A build_embeddings function creates random 4-dimensional embeddings for each word in a given sentence. A fixed random seed (500) ensures reproducibility.

Self-Attention Calculation:
The self_attention function:

Constructs Query (Q), Key (K), and Value (V) matrices.

Computes attention scores using the dot product of Q and K.

Applies the softmax function to obtain attention weights.

Generates weighted embeddings by combining the attention weights with the V matrix.

Analysis and Output:
The script prints:

The original sentence.

For each word, the indices and corresponding attention weights of the top three words that it attends to.

An explanation of which word (other than itself) receives the second-highest attention.

Installation & Usage
Requirements
Python 3.7+

PyTorch

TensorFlow (for the Seq2Seq translation)

Transformers

Datasets

NumPy

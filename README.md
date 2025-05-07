# Transformer & Sequence Model Showcase

This repository compiles multiple code examples that demonstrate how to use and fine-tune several Transformer models and sequence learning techniques. The projects include:

- **BERT Fine-Tuning** for sequence classification  
- **GPT-2 Fine-Tuning** for causal language modeling  
- **GLUE MRPC Data Processing** for tokenization and collation  
- **LangTranslator**, featuring two translation approaches:  
  - T5-based translation using conditional generation  
  - Sequence-to-Sequence translation with LSTM (TensorFlow/Keras)  
- **Self-Attention Mechanism** demonstrated using NumPy  

---

## Table of Contents

1. [BERT Fine-Tuning](#bert-fine-tuning)  
2. [GPT-2 Fine-Tuning](#gpt-2-fine-tuning)  
3. [GLUE MRPC Data Processing](#glue-mrpc-data-processing)  
4. [LangTranslator](#langtranslator)  
   - [T5-based Translation](#t5-based-translation)  
   - [Seq2Seq Translation with LSTM](#seq2seq-translation-with-lstm)  
5. [Self-Attention Mechanism](#self-attention-mechanism)  
6. [Installation & Usage](#installation--usage)  
7. [License](#license)  

---

## BERT Fine-Tuning

This project demonstrates how to fine-tune the `bert-base-uncased` model for a sequence classification task using Hugging Face's Transformers library and PyTorch.

**Overview**  
- **Imports Required Libraries:** Integrates the Transformers library with PyTorch.  
- **Loads Pretrained Model and Tokenizer:** Uses `BertTokenizer` and `BertForSequenceClassification` with `num_labels=2`.  
- **Defines a Custom Dataset:**  
  - Tokenizes, pads/truncates to 128 tokens, and converts texts & labels to tensors.  
- **Prepares Sample Data:** Hardcoded training & evaluation splits.  
- **Training Configuration & Execution:** Uses `TrainingArguments` and `Trainer`, then saves model & tokenizer locally.  

---

## GPT-2 Fine-Tuning

This example fine-tunes the GPT-2 model for a causal language modeling task.

**Overview**  
- **Loading Model & Tokenizer:** Uses `AutoModelForCausalLM` and `AutoTokenizer`.  
- **Preparing the Dataset:** Writes sample paragraphs to a temp file, loads via `TextDataset`.  
- **Data Collation:** Uses `DataCollatorForLanguageModeling` (no masking).  
- **Training & Saving:** Configures with `TrainingArguments` and `Trainer`, then saves both model & tokenizer.  

---

## GLUE MRPC Data Processing

Demonstrates loading, tokenizing, and collating the GLUE MRPC dataset using Hugging Face Datasets and Transformers.

**Overview**  
- **Dataset Loading:** Loads GLUE MRPC via the Datasets library.  
- **Tokenization:** Applies `bert-base-uncased` tokenizer for single and paired sentences, and converts IDs back to tokens.  
- **Collation:** Uses a padding collator to batch tokenized samples.  
- **Output:** Saves collated tensors as `glue_mrpc_labled_tensor.pt`.  

---

## LangTranslator

Two approaches to language translation are provided:

### T5-based Translation

Uses a pre-trained T5 model (e.g., `t5-small`) for conditional generation.

**Overview**  
- **Model & Tokenizer Setup:** Loads `T5ForConditionalGeneration` & `T5Tokenizer`.  
- **Translation Function:** Formats prompts, tokenizes, generates, and decodes translations.  
- **Example:** Prints tokenized inputs and translated output.  

### Seq2Seq Translation with LSTM

Implements a TensorFlow/Keras encoder–decoder with LSTM layers.

**Overview**  
- **Data Preparation:** Tokenizes input/target pairs, adds `<sos>`/`<eos>`, and pads.  
- **Model Definition:** Embedding + LSTM encoder; LSTM decoder using encoder state.  
- **Training & Inference:** Trains with sparse categorical cross-entropy and Adam; inference generates tokens until `<eos>`.  

---

## Self-Attention Mechanism

A NumPy-based self-attention example.

**Overview**  
- **Embedding Generation:** `build_embeddings` creates reproducible 4‑dim vectors per word.  
- **Self-Attention Calculation:**  
  - Builds Q, K, V matrices.  
  - Computes dot-product scores, applies softmax, and weights V to get attended embeddings.  
- **Analysis & Output:** Prints the original sentence, top‑3 attention indices/weights per word, and identifies the second‑highest attended word.  

---

## Installation & Usage

**Requirements**  
- Python 3.7+  
- PyTorch  
- TensorFlow (for Seq2Seq)  
- Transformers  
- Datasets  
- NumPy  

```bash
pip install -r requirements.txt

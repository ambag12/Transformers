from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import torch
raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
inputs = tokenizer("This is the first sentence.", "This is the second one.")
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
batch = data_collator(samples)
newsample={k: v.shape for k, v in batch.items()}
print(batch)

torch.save(batch, "glue_mrpc_labled_tensor.pt")
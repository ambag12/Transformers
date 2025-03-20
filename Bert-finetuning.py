from transformers import BertTokenizerFast,BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TextDataset,DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.column_names = ["input_ids", "attention_mask", "labels"]  
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

texts = [
    "Apple released a new iPhone.",
    "Google launched a new search feature.",
    "Tesla introduced an electric vehicle.",
    "Amazon profits increased.",
    "Microsoft updated its operating system."
]
labels = [1, 0, 1, 0, 1]  
train_texts, eval_texts = texts[:3], texts[3:]
train_labels, eval_labels = labels[:3], labels[3:]

train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
eval_dataset = CustomDataset(tokenizer, eval_texts, eval_labels)

training_args = TrainingArguments(
    output_dir="./results",  
    num_train_epochs=3,      
    per_device_train_batch_size=2,  
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
)

trainer.train()
print("Fine-tuning complete.")
model.save_pretrained("fine-tuned-bert")
tokenizer.save_pretrained("fine_tuned_bert")
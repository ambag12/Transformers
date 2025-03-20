from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import tempfile


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

text_data = """Apple Inc. released its new iPhone model yesterday, featuring a longer battery life and a more advanced camera system. Experts say the release positions Apple as a strong contender in the upcoming holiday shopping season.
Google announced a new feature in its search engine that aims to combat misinformation. The feature will prompt users with reliable sources when they search for topics that are prone to misinformation.
Tesla's latest electric vehicle model has been receiving rave reviews for its performance and sustainability features. However, critics point out that the high price tag may make it inaccessible for many consumers.
Amazon's quarterly profits exceeded expectations, boosted by the surge in online shopping amid the COVID-19 pandemic. The company announced that it would be hiring 100,000 more workers to keep up with demand.
Microsoft's new operating system update includes several features aimed at enhancing user privacy. The update will be rolled out to users in phases starting next month. """
with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
    tmp_file.write(text_data)
    file_path = tmp_file.name

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_path,  # Use the temporary file path
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
print(trainer)

model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")
import os
os.environ["WANDB_MODE"] = "disabled"  # Disable Weights & Biases logging

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

class DescriptionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                title = item.get("title", "").strip()
                features = item.get("features", [])
                description = item.get("description", "").strip()

                if not title or not features or not description:
                    continue

                features_text = ", ".join(features)
                prompt = f"Title: {title}\nFeatures: {features_text}\nDescription:"
                full_text = prompt + " " + description

                encoding = tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoding.input_ids.squeeze()
                attention_mask = encoding.attention_mask.squeeze()

                prompt_encoding = tokenizer(
                    prompt,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                prompt_len = (prompt_encoding.input_ids != tokenizer.pad_token_id).sum()

                labels = input_ids.clone()
                labels[:prompt_len] = -100

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    dataset_file = r"/content/meta_Handmade_Products_cleaned.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    full_dataset = DescriptionDataset(dataset_file, tokenizer, max_length=512)
    print(f"Total dataset size: {len(full_dataset)}")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="./lora_finetuned_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=3e-4,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("./lora_finetuned_model")
    tokenizer.save_pretrained("./lora_finetuned_model")

    print("✅ Training complete and model saved at ./lora_finetuned_model")
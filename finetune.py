import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSON data
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                question = item["question"]
                answer = item["answer_cot"]
                # Combine question and answer for causal language modeling
                text = f"Question: {question}\nAnswer: {answer}"
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.data[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }

def fine_tune(model_name, train_file, output_dir, epochs=3, batch_size=4):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset
    dataset = CustomDataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none",
        push_to_hub=False
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    # Define parameters
    MODEL_NAME = "gpt2"
    TRAIN_FILE = "data/train_data.json"
    OUTPUT_DIR = "./fine_tuned_model"
    EPOCHS = 3
    BATCH_SIZE = 2

    # Fine-tune the model
    fine_tune(MODEL_NAME, TRAIN_FILE, OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
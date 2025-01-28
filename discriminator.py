import json
import random
import torch
import platform
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F

class ClassificationDataset(Dataset):
    def __init__(self, file_1, file_2, tokenizer, max_length=512):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data from file_1 with label 0
        with open(file_1, 'r') as f1:
            raw_data_1 = json.load(f1)
            for item in raw_data_1:
                question = item["question"]
                answer = item["answer_cot"]
                text = f"Question: {question}\nAnswer: {answer}"
                self.data.append(text)
                self.labels.append(0)

        # Load data from file_2 with label 1
        with open(file_2, 'r') as f2:
            raw_data_2 = json.load(f2)
            for item in raw_data_2:
                question = item["question"]
                answer = item["answer_cot"]
                text = f"Question: {question}\nAnswer: {answer}"
                self.data.append(text)
                self.labels.append(1)

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
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

class RewardModel(torch.nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, 1)  # Binary classification
        self.device = device

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # Use the last hidden state (corresponds to [batch_size, seq_len, hidden_size])
        hidden_state = outputs.hidden_states[-1][:, 0, :]  # Take the CLS token representation
        logits = self.classifier(hidden_state).squeeze(-1)  # Reduce to [batch_size]

        if labels is not None:
            # Compute binary cross-entropy loss
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits

def fine_tune_classification(model_name, train_file_1, train_file_2, output_dir, epochs=3, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = ClassificationDataset(train_file_1, train_file_2, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RewardModel(model_name, device).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

def test_reward_model(model, tokenizer, input_text, device):
    """
    Test the fine-tuned reward model with a given text input.

    Args:
        model: The fine-tuned reward model.
        tokenizer: The tokenizer used for preprocessing text.
        input_text: The text input to classify.
        device: The device on which the model is loaded.

    Returns:
        A dictionary with class probabilities.
    """
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input text
    encoded_input = tokenizer(
        input_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    return {
        "TRAIN_FILE_1": 1 - probabilities[0],  # Probability for class 0
        "TRAIN_FILE_2": probabilities[0]       # Probability for class 1
    }

if __name__ == "__main__":
    # Define parameters
    MODEL_NAME = "gpt2"
    TRAIN_FILE_1 = "data/train_data_modified.json"
    TRAIN_FILE_2 = "data/train_data_fallacy_modified.json"
    OUTPUT_DIR = "./fine_tuned_reward_model"
    EPOCHS = 3
    BATCH_SIZE = 4

    # Fine-tune the model
    fine_tune_classification(MODEL_NAME, TRAIN_FILE_1, TRAIN_FILE_2, OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
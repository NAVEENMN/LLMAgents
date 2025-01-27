import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_test_data(file_path):
    """Load test data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_answer_from_cot(cot):
    """
    Extract the final numeric answer enclosed in < > from the chain of thought (answer_cot).
    Returns None if no such answer is found.
    """
    match = re.search(r"<(\d+)>", cot)
    if match:
        return float(match.group(1))
    else:
        # Optionally, try extracting standalone numbers as a fallback
        number_match = re.findall(r"\d+", cot)
        if number_match:
            return float(number_match[-1])  # Return the last number found as a fallback
    return None

def evaluate(model, tokenizer, test_data, max_new_tokens=100, temperature=0.5):
    """Evaluate the model on the test dataset."""
    correct_predictions = 0
    total_samples = len(test_data)
    missing_predictions = 0  # Count cases where no answer is extracted

    for item in test_data:
        question = item["question"]
        ground_truth = float(item["answer_value"])  # Convert ground truth to float
        
        # Generate answer_cot using the model
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )
        generated_cot = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the predicted answer
        predicted_answer = extract_answer_from_cot(generated_cot)
        
        # Handle missing predictions
        if predicted_answer is None:
            missing_predictions += 1
            print(f"Missing answer for question: {question}")
        elif predicted_answer == ground_truth:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    missing_rate = missing_predictions / total_samples

    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Missing Predictions: {missing_predictions} ({missing_rate * 100:.2f}%)")
    return accuracy

if __name__ == "__main__":
    # Paths and model name
    TEST_FILE = "data/test_data.json"
    MODEL_PATH = "./fine_tuned_model"
    test_data = load_test_data(TEST_FILE)

    # Load fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate the model
    accuracy = evaluate(model, tokenizer, test_data)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
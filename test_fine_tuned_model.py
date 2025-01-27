import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer from the specified path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    """
    Generate a response from the fine-tuned model given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Path to the fine-tuned model directory
    MODEL_PATH = "fine_tuned_model"

    # Load the model and tokenizer
    print("Loading the fine-tuned model...")
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
    print("Model loaded successfully!")

    # Interactive prompt
    print("\nInteractive Mode: Enter your prompt below.")
    print("Type 'exit' to quit.\n")
    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() == "exit":
            print("Exiting...")
            break

        # Generate response
        response = generate_response(model, tokenizer, prompt)
        print("\nGenerated Response:")
        print(response)
        print("-" * 80)
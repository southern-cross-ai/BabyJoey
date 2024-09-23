import torch
from transformers import GPT2Tokenizer
from model import BabyJoey 

def generate_text(model, tokenizer, start_token="<|startoftext|>", max_length=50, temperature=0.7, device="cpu"):
    """
    Generate a sequence of text using the trained model.
    
    Args:
        model: The trained BabyJoey model.
        tokenizer: The tokenizer used to encode/decode text.
        start_token: The initial token to start generation.
        max_length: The maximum length of the generated sequence.
        temperature: Sampling temperature for controlling randomness.
        device: Device to run the model on ('cpu' or 'cuda').
    
    Returns:
        generated_text: The generated sequence in text form.
    """
    model.eval()  # Set the model to evaluation mode
    start_token_ids = tokenizer.encode(start_token, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        generated_sequence = model.generate(input_ids=start_token_ids, max_length=max_length, temperature=temperature)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_sequence[0].tolist(), skip_special_tokens=True)
    
    return generated_text

def main():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the trained model
    model = BabyJoey() 
    model.load_state_dict(torch.load("baby_joey_model.pth"))
    model.to(device)
    
    # Generate text
    generated_text = generate_text(model, tokenizer, start_token="<|startoftext|>", max_length=50, device=device)
    
    # Print the generated text
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
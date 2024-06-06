import re
import json
from collections import Counter

class BabyJoeyTokenizer:
    def __init__(self, special_tokens=["<pad>", "<unk>", "<s>", "</s>"], vocab_size=10000):
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, texts):
        # Tokenize texts
        tokens = [self.tokenize(text) for text in texts]
        
        # Flatten token list
        all_tokens = [token for sublist in tokens for token in sublist]
        
        # Count frequencies
        token_freqs = Counter(all_tokens)
        
        # Select most common tokens
        most_common_tokens = [token for token, _ in token_freqs.most_common(self.vocab_size - len(self.special_tokens))]
        
        # Create vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens + most_common_tokens)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def tokenize(self, text):
        # Simple whitespace tokenizer, you can customize this
        return re.findall(r'\b\w+\b', text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(token_id, "<unk>") for token_id in token_ids])

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
            self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

if __name__ == "__main__":
    # Example usage
    tokenizer = CustomTokenizer()
    texts = [
        "This is an example sentence.",
        "This is another sentence."
    ]
    
    # Build vocabulary
    tokenizer.build_vocab(texts)
    
    # Save vocabulary
    tokenizer.save_vocab("vocab.json")

    # Encode text
    encoded = tokenizer.encode("This is an example sentence.")
    print(f"Encoded: {encoded}")
    
    # Decode text
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

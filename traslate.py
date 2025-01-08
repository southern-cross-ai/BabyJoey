import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# Configurations
BATCH_SIZE = 1
MAX_SEQ_LEN = 128
D_MODEL = 512
NHEAD = 2
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("opus_books", "en-fr")

tokenizer_src = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_tgt = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize dataset
def tokenize_function(examples):
    src_texts = [ex["en"] for ex in examples["translation"]]
    tgt_texts = [ex["fr"] for ex in examples["translation"]]

    src = tokenizer_src(src_texts, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)
    tgt = tokenizer_tgt(tgt_texts, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

    return {
        "input_ids": src["input_ids"],
        "labels": tgt["input_ids"],
    }

# Handle missing validation split
if "validation" not in dataset.keys():
    dataset = dataset["train"].train_test_split(test_size=0.1)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["translation"])
tokenized_datasets.set_format(type="torch", columns=["input_ids", "labels"])

# Create DataLoaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

# Define Transformer Model
class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * (D_MODEL ** 0.5)
        tgt = self.tgt_embedding(tgt) * (D_MODEL ** 0.5)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        tgt = tgt.permute(1, 0, 2)
        out = self.transformer(src, tgt)
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        return self.fc_out(out)

# Model Initialization
src_vocab_size = tokenizer_src.vocab_size
tgt_vocab_size = tokenizer_tgt.vocab_size
model = TranslationTransformer(
    src_vocab_size, tgt_vocab_size, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
).to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# Function to generate a translation

def generate_translation(model, tokenizer_src, tokenizer_tgt, input_text, max_len=50):
    model.eval()
    src_tokens = tokenizer_src(input_text, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True, padding="max_length")[
        "input_ids"
    ].to(DEVICE)
    tgt_tokens = torch.tensor([[tokenizer_tgt.cls_token_id]], device=DEVICE)  # Start with <CLS> token for target

    for _ in range(max_len):
        output = model(src_tokens, tgt_tokens)
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Get token with highest probability
        tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

        if next_token.item() == tokenizer_tgt.sep_token_id:
            break

    return tokenizer_tgt.decode(tgt_tokens.squeeze().tolist(), skip_special_tokens=True)

# Training Loop
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        src = batch["input_ids"].to(DEVICE)
        tgt = batch["labels"].to(DEVICE)
        tgt_input = tgt[:, :-1]  # Remove the last token for target input
        tgt_output = tgt[:, 1:]  # Shift by one for target output

        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation Loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            src = batch["input_ids"].to(DEVICE)
            tgt = batch["labels"].to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(valid_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

    # Save the model
    epoch_save_path = f"model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), epoch_save_path)
    print(f"Model saved at {epoch_save_path}")

    # Test Sentence Translation
    test_sentence = "This is a test sentence for translation."
    translation = generate_translation(model, tokenizer_src, tokenizer_tgt, test_sentence)
    print(f"Test Sentence: {test_sentence}")
    print(f"Translation: {translation}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from byte_converter import TokenToByteConverter
from byte_model import ByteStudentModel
from loss_utils import DistillationLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load teacher model and tokenizer
    print("Loading teacher model (GPT-2)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    teacher.eval()

    converter = TokenToByteConverter(tokenizer)
    
    seq_len = 64
    student = ByteStudentModel(vocab_size=256, d_model=256, n_layers=4, n_heads=4).to(device)
    criterion = DistillationLoss(temperature=2.0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # Load local sample text
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True, padding="max_length")
    input_ids = encodings["input_ids"].to(device)
    byte_tokens = torch.tensor([[ord(c) for c in text[:seq_len].ljust(seq_len, '\0')]], dtype=torch.long).to(device)

    print("\nStarting training loop...")
    student.train()
    
    epochs = 50
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher(input_ids)
            teacher_token_logits = teacher_outputs.logits

        teacher_byte_logits = converter.convert_logits(teacher_token_logits)
        student_logits = student(byte_tokens)

        loss = criterion(student_logits, teacher_byte_logits)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    # Save trained checkpoint
    torch.save(student.state_dict(), "byte_student_checkpoint.pt")
    print("\nTraining complete. Checkpoint saved to byte_student_checkpoint.pt")

if __name__ == "__main__":
    main()
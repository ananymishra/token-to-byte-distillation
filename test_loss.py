import torch
from loss_utils import DistillationLoss

def main():
    print("Initializing distillation loss...")
    criterion = DistillationLoss(temperature=2.0)

    # Dummy student and teacher logits: [batch, seq_len, vocab_size]
    student_logits = torch.randn(2, 10, 256, requires_grad=True)
    teacher_logits = torch.randn(2, 10, 256)

    print("Computing KL divergence loss...")
    loss = criterion(student_logits, teacher_logits)

    print("\n--- Loss Test Results ---")
    print(f"Loss value: {loss.item():.4f}")
    
    # Verify backpropagation works smoothly
    loss.backward()
    print("Backward pass successful. Gradients computed.")

if __name__ == "__main__":
    main()
import torch
from byte_model import ByteStudentModel

def main():
    print("Initializing student byte model...")
    model = ByteStudentModel(vocab_size=256, d_model=128, n_layers=2, n_heads=2)

    # Batch of 2 sequences, each 10 bytes long
    dummy_byte_ids = torch.randint(0, 256, (2, 10))

    print("Passing byte sequence through model...")
    logits = model(dummy_byte_ids)

    print("\n--- Test Results ---")
    print(f"Input shape:  {dummy_byte_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("Verification complete.")

if __name__ == "__main__":
    main()
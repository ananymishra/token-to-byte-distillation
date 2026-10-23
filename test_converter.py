import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from byte_converter import TokenToByteConverter

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

print("Initializing token-to-byte converter...")
converter = TokenToByteConverter(tokenizer)

# Test text
text = "Hello, world! Byte distillation is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    token_logits = outputs.logits

# Convert 50,257 token logits to 256 byte logits
byte_logits = converter.convert_logits(token_logits)

print("\n--- Conversion Check ---")
print("Original token logits shape:", token_logits.shape)
print("Converted byte logits shape:", byte_logits.shape)
print("Success! 256-dimensional byte logits generated.")
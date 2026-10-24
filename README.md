# Breaking the Token Ceiling: Micro-Scale Byte Distillation

An end-to-end PyTorch implementation of **End-Of-Token (EOT) Cross-Vocabulary Knowledge Distillation** (adapted from *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models*).

## Project Roadmap
- [x] **Phase 0:** Environment & CUDA Verification (RTX 3070 Ti)
- [x] **Phase 1:** Vectorized Token-to-Byte (`<EOT>`) Logit Converter (`byte_converter.py` & `test_converter.py`)
- [x] **Phase 2:** Lightweight Student Transformer Model & Distillation Loss (`byte_model.py`, `loss_utils.py`, and test scripts)
- [ ] **Phase 3:** Offline Byte Logit Caching (Storage Compression Benchmark)
- [ ] **Phase 4:** Student Training (`Token-Student` vs. `EOT-Byte-Student`)
- [ ] **Phase 5:** Character-Noise Robustness Evaluation & Interactive Demo

## Project Structure
- **`byte_converter.py`**: Projects GPT-2's `50,257` token logits down to `256` byte logits via matrix multiplication.
- **`byte_model.py`**: A lightweight PyTorch Transformer encoder featuring causal masking and pre-layer normalization.
- **`loss_utils.py`**: Temperature-scaled Kullback-Leibler (KL) Divergence loss for soft-target knowledge distillation.
- **Verification Scripts**: Test suites validating shape transformations, forward passes, and gradient computations.

## Tech Stack
- Python 3.12
- PyTorch
- Hugging Face Transformers
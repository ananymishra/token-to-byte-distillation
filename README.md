# Breaking the Token Ceiling: Micro-Scale Byte Distillation

An end-to-end PyTorch implementation of **End-Of-Token (EOT) Cross-Vocabulary Knowledge Distillation** (adapted from *Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models*).

## Project Roadmap
- [x] **Phase 0:** Environment & CUDA Verification (RTX 3070 Ti)
- [ ] **Phase 1:** Vectorized Token-to-Byte (`<EOT>`) Logit Converter
- [ ] **Phase 2:** Offline Byte Logit Caching (Storage Compression Benchmark)
- [ ] **Phase 3:** Student Training (`Token-Student` vs. `EOT-Byte-Student`)
- [ ] **Phase 4:** Character-Noise Robustness Evaluation & Interactive Demo
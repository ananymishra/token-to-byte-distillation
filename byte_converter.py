import torch

class TokenToByteConverter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.mapping_matrix = self._build_mapping_matrix()

    def _build_mapping_matrix(self):
        # [vocab_size, 256] grid: maps each token ID to its ending byte
        matrix = torch.zeros((self.vocab_size, 256), dtype=torch.float32)
        
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            
            # Convert the token string back to its raw text/byte representation
            decoded_text = self.tokenizer.convert_tokens_to_string([token_str])
            token_bytes = decoded_text.encode('utf-8', errors='ignore')
            
            if len(token_bytes) > 0:
                # Mark 1.0 at the column corresponding to the token's final byte
                last_byte = token_bytes[-1]
                matrix[token_id, last_byte] = 1.0
                
        return matrix

    def convert_logits(self, token_logits):
        # Project token logits [batch, seq, vocab] down to byte logits [batch, seq, 256]
        device = token_logits.device
        matrix = self.mapping_matrix.to(device)
        return torch.matmul(token_logits, matrix)
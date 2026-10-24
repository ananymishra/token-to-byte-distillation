import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # batchmean standardizes the KL divergence across batch and sequence dimensions
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        # Scale logits by temperature to soften probability distributions
        scaled_student = student_logits / self.temperature
        scaled_teacher = teacher_logits / self.temperature

        # KLDivLoss requires log-probabilities for student (q) and probabilities for teacher (p)
        log_q = F.log_softmax(scaled_student, dim=-1)
        p = F.softmax(scaled_teacher, dim=-1)

        # Scale loss by T^2 to maintain gradient magnitude scaling across temperatures
        return self.kl_div(log_q, p) * (self.temperature ** 2)
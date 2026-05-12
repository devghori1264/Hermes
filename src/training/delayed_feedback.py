import numpy as np

class DelayedFeedbackEstimator:
    def __init__(self, lambda_delay: float = 0.1) -> None:
        self.lambda_delay = lambda_delay

    def compute_survival_probability(self, elapsed_time: np.ndarray) -> np.ndarray:
        return np.exp(-self.lambda_delay * elapsed_time)

    def fsiw_weights(self, elapsed_time: np.ndarray, is_positive: np.ndarray) -> np.ndarray:
        survival_prob = self.compute_survival_probability(elapsed_time)
        survival_prob = np.clip(survival_prob, 1e-6, 1.0)
        
        weights = np.ones_like(is_positive, dtype=np.float32)
        negative_mask = (is_positive <= 0)
        
        weights[negative_mask] = 1.0 / survival_prob[negative_mask]
        return weights

    def apply_loss_correction(self, raw_loss: np.ndarray, elapsed_time: np.ndarray, is_positive: np.ndarray) -> np.ndarray:
        weights = self.fsiw_weights(elapsed_time, is_positive)
        return raw_loss * weights

class ModelDistiller:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        self.temperature = temperature
        self.alpha = alpha

    def compute_distillation_loss(
        self, 
        student_logits: np.ndarray, 
        teacher_logits: np.ndarray, 
        true_labels: np.ndarray
    ) -> float:
        student_probs = self._softmax(student_logits / self.temperature)
        teacher_probs = self._softmax(teacher_logits / self.temperature)
        
        soft_loss = -np.sum(teacher_probs * np.log(student_probs + 1e-12)) * (self.temperature ** 2)
        
        student_hard_probs = self._softmax(student_logits)
        hard_loss = -np.sum(true_labels * np.log(student_hard_probs + 1e-12))
        
        return float(self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

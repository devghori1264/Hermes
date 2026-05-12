import torch
import torch.optim as optim
from typing import Iterator
from src.ranking.deep_models import xDeepFM
from src.ranking.losses import listwise_listmle_loss

class DeepRankingTrainer:
    def __init__(self, model: xDeepFM, learning_rate: float = 0.001) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_step(self, features: torch.Tensor, labels: torch.Tensor, batch: dict = None) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        targets = labels.clone().detach()

        if batch is not None:
            try:
                from src.training.delayed_feedback import DelayedFeedbackEstimator
                if "elapsed_time" in batch and batch["elapsed_time"] is not None:
                    estimator = DelayedFeedbackEstimator()
                    elapsed = batch["elapsed_time"].numpy()
                    is_pos = targets.numpy()
                    weights = estimator.fsiw_weights(elapsed, is_pos)
                    targets = targets * torch.tensor(weights, dtype=torch.float32)
            except ImportError:
                pass
        
        scores = self.model(features)
        
        scores_list = scores.tolist()
        labels_list = targets.tolist()
        
        loss_result = listwise_listmle_loss(scores_list, labels_list)
        loss_val = torch.tensor(loss_result.value, requires_grad=True)
        
        loss_val.backward()
        self.optimizer.step()
        
        return float(loss_val.item())

    def train_epoch(self, dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> float:
        total_loss = 0.0
        batches = 0
        for features, labels in dataloader:
            loss = self.train_step(features, labels)
            total_loss += loss
            batches += 1
        return total_loss / max(1, batches)

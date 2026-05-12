import hashlib
from typing import Dict, Optional
from pydantic import BaseModel

class ExperimentGroup(BaseModel):
    experiment_id: str
    group_name: str
    model_version: str

class ABTestingFramework:
    def __init__(self, salt: str = "hermes_salt_2026"):
        self.salt = salt
        self.experiments: Dict[str, Dict[str, float]] = {
            "ranking_v2_test": {
                "control": 0.5,
                "treatment_a": 0.5
            }
        }
        self.model_mapping = {
            "ranking_v2_test": {
                "control": "xdeepfm_v1",
                "treatment_a": "two_tower_v2"
            }
        }

    def _hash_user(self, user_id: str, experiment_id: str) -> float:
        payload = f"{user_id}_{experiment_id}_{self.salt}".encode('utf-8')
        digest = hashlib.md5(payload).hexdigest()
        return int(digest[:8], 16) / 0xffffffff

    def assign_group(self, user_id: Optional[str], experiment_id: str) -> ExperimentGroup:
        if not user_id or experiment_id not in self.experiments:
            return ExperimentGroup(
                experiment_id=experiment_id,
                group_name="control",
                model_version="default"
            )

        hash_val = self._hash_user(user_id, experiment_id)
        
        cumulative = 0.0
        for group, weight in self.experiments[experiment_id].items():
            cumulative += weight
            if hash_val <= cumulative:
                return ExperimentGroup(
                    experiment_id=experiment_id,
                    group_name=group,
                    model_version=self.model_mapping[experiment_id].get(group, "default")
                )
                
        return ExperimentGroup(
            experiment_id=experiment_id,
            group_name="control",
            model_version="default"
        )

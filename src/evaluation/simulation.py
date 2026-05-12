import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SimulatedUser:
    user_id: str
    latent_preferences: np.ndarray
    domain_affinity: Dict[str, float]

@dataclass
class SimulatedEvent:
    user_id: str
    item_id: str
    event_type: str
    timestamp: float

class UserBehaviorSimulator:
    def __init__(self, num_users: int = 100, latent_dim: int = 384):
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.users: Dict[str, SimulatedUser] = {}
        self._initialize_users()
        
    def _initialize_users(self):
        domains = ["movies", "music", "news", "commerce"]
        for i in range(self.num_users):
            uid = f"sim_user_{i}"
            # Dirichlet prior for domain affinity
            affinities = np.random.dirichlet(np.ones(len(domains)))
            domain_map = {d: float(a) for d, a in zip(domains, affinities)}
            
            # Latent preference vector matching Faiss dimension
            latent = np.random.randn(self.latent_dim)
            latent = latent / np.linalg.norm(latent)
            
            self.users[uid] = SimulatedUser(uid, latent, domain_map)

    def simulate_session(self, recommender_service, user_id: str, current_time: float) -> List[SimulatedEvent]:
        user = self.users[user_id]
        events = []
        
        # Pick dominant domain for this session
        target_domain = max(user.domain_affinity.items(), key=lambda x: x[1])[0]
        
        # In a real simulation we'd pass the user's latent vector directly to retrieve candidates.
        # Here we mock it by querying a generic seed related to the domain.
        seeds = {"movies": "Inception", "music": "Beatles", "news": "Technology", "commerce": "Laptop"}
        seed_query = seeds.get(target_domain, "Matrix")
        
        # Recommender generates ranked items
        from src.domain.models import RecommendationContext
        ctx = RecommendationContext(query_item_title=seed_query, user_id=user_id)
        
        # Try-except block to handle uninitialized FAISS states during simulation
        try:
            ranked_items = recommender_service.recommend_ranked(seed_query, context=ctx)
        except Exception:
            return events
            
        # Simulate clicks based on position bias and latent match
        for idx, item in enumerate(ranked_items):
            # Position bias: decays exponentially
            position_bias = np.exp(-0.5 * idx)
            
            # Base click prob
            click_prob = position_bias * 0.4
            
            # Record impression
            events.append(SimulatedEvent(user_id, item.item_id, "impression", current_time))
            
            if np.random.random() < click_prob:
                events.append(SimulatedEvent(user_id, item.item_id, "click", current_time + 5.0))
                
                # Conditional purchase/watch
                if np.random.random() < 0.2:
                    events.append(SimulatedEvent(user_id, item.item_id, "conversion", current_time + 60.0))
                    
        return events

def run_simulation(recommender_service, num_sessions: int = 50, output_path: str = "benchmark_results.csv"):
    import pandas as pd
    simulator = UserBehaviorSimulator()
    all_events = []
    
    current_time = 1700000000.0
    for i in range(num_sessions):
        user_id = f"sim_user_{np.random.randint(0, simulator.num_users)}"
        events = simulator.simulate_session(recommender_service, user_id, current_time)
        all_events.extend(events)
        current_time += np.random.exponential(3600)  # Next session arrival
        
    df = pd.DataFrame([{
        "user_id": e.user_id, 
        "item_id": e.item_id, 
        "event_type": e.event_type, 
        "timestamp": e.timestamp
    } for e in all_events])
    
    df.to_csv(output_path, index=False)
    print(f"Simulation complete. Generated {len(all_events)} events. Saved to {output_path}")
    return all_events

if __name__ == "__main__":
    from src.services.recommendation_service import RecommendationService
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent.parent
    recommender = RecommendationService(base_path / "main_data.csv")
    run_simulation(recommender, num_sessions=10, output_path=str(base_path / "Plan" / "benchmark_simulation.csv"))

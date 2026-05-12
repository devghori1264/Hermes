import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_user_features: int, num_item_features: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        
        user_layers = []
        in_dim = num_user_features
        for h_dim in hidden_dims:
            user_layers.append(nn.Linear(in_dim, h_dim))
            user_layers.append(nn.ReLU())
            in_dim = h_dim
        user_layers.append(nn.Linear(in_dim, output_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        item_layers = []
        in_dim = num_item_features
        for h_dim in hidden_dims:
            item_layers.append(nn.Linear(in_dim, h_dim))
            item_layers.append(nn.ReLU())
            in_dim = h_dim
        item_layers.append(nn.Linear(in_dim, output_dim))
        self.item_tower = nn.Sequential(*item_layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_tower(user_features)
        i_emb = self.item_tower(item_features)
        return torch.sum(u_emb * i_emb, dim=1)

class CIN(nn.Module):
    def __init__(self, num_fields: int, layer_sizes: list[int]) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.layer_sizes = layer_sizes
        self.conv_layers = nn.ModuleList()
        prev_size = num_fields
        for size in layer_sizes:
            self.conv_layers.append(nn.Conv1d(self.num_fields * prev_size, size, 1))
            prev_size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_fields, embed_dim = x.shape
        hidden_states = [x]
        x_0 = x.unsqueeze(2)
        
        for i, size in enumerate(self.layer_sizes):
            x_k = hidden_states[-1].unsqueeze(1)
            outer = torch.matmul(x_0, x_k)
            outer = outer.view(batch_size, self.num_fields * hidden_states[-1].shape[1], embed_dim)
            curr_state = F.relu(self.conv_layers[i](outer))
            hidden_states.append(curr_state)

        result = torch.cat([h.sum(dim=2) for h in hidden_states[1:]], dim=1)
        return result

class xDeepFM(nn.Module):
    def __init__(self, num_features: int, embed_dim: int, dnn_hidden_dims: list[int], cin_layer_sizes: list[int]) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(num_features, num_features * embed_dim)
        self.num_fields = num_features
        
        self.cin = CIN(self.num_fields, cin_layer_sizes)
        
        dnn_layers = []
        in_dim = num_features * embed_dim
        for h_dim in dnn_hidden_dims:
            dnn_layers.append(nn.Linear(in_dim, h_dim))
            dnn_layers.append(nn.ReLU())
            in_dim = h_dim
        self.dnn = nn.Sequential(*dnn_layers)
        
        self.dnn_out = nn.Linear(in_dim, 1)
        self.cin_out = nn.Linear(sum(cin_layer_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_term = self.linear(x)
        
        emb = self.embedding(x)
        emb = emb.view(x.shape[0], self.num_fields, self.embed_dim)
        
        cin_term = self.cin_out(self.cin(emb))
        
        dnn_term = self.dnn_out(self.dnn(emb.view(x.shape[0], -1)))
        
        return linear_term.squeeze(1) + cin_term.squeeze(1) + dnn_term.squeeze(1)

class MLP(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.C = embedding_dim
    self.MLP = nn.Sequential( nn.Linear(self.C, 4*self.C) , nn.GELU() , nn.Linear(4*self.C, self.C))
  def forward(self,E):
    return self.MLP(E)

class MoE_block(nn.Module):
  def __init__(self, context_length, embedding_dim, num_of_experts, active_experts, expert_capacity_factor=1.25):
    super().__init__()
    self.T = context_length
    self.C = embedding_dim
    self.N = num_of_experts
    self.k = active_experts
    self.cap_factor = expert_capacity_factor
    self.experts = nn.ModuleList([MLP(self.C) for expert in range(self.N)])
    self.router = nn.Linear(self.C, self.N)

  def forward(self, E):
    B,T,C = E.shape
    E_flat = E.reshape(B*T, C)
    logits = self.router(E_flat) # now (B*T, N)
    prob_dist = F.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(prob_dist, k=self.k, dim=-1) # now (B*T, k)
    max_tokens_per_expert = int((B*T*self.k) / self.N*self.cap_factor)
    expert_map = F.one_hot(topk_idx, num_classes=self.N).float() # now (B*T, k, N)
    expert_counts = torch.cumsum(expert_map, dim=0)
    positions = (expert_counts * expert_map).sum(dim=-1).long()-1 # get matrix where positions are expert assignment and num is num of token going into that expert, now (tokens,k)
    mask = positions < max_tokens_per_expert
    topk_vals = topk_vals * mask
    expert_map = expert_map * mask.unsqueeze(-1)
    dispatch = torch.zeros(B*T, self.N, max_tokens_per_expert, device=E.device)
    token_idx = torch.arange(B*T).unsqueeze(1).expand(-1,self.k)
    safe_positions = positions.clamp(max=max_tokens_per_expert-1)
    dispatch[token_idx, topk_idx, safe_positions] = 1
    expert_inputs = torch.einsum("td,tnc->ncd", E_flat, dispatch) # now (N, max_tokens_per_expert, C)
    expert_outputs = []
    for i in range(self.N):
      expert_outputs.append(self.experts[i](expert_inputs[i]))
    expert_outputs = torch.stack(expert_outputs, dim=0)
    combine = torch.zeros_like(dispatch)
    combine[token_idx, topk_idx, safe_positions] = topk_vals
    output = torch.einsum("ncd,tnc->td", expert_outputs, combine)
    f_i = expert_map.sum(dim=(0,1)) / B*T
    p_i = prob_dist.mean(dim=0)
    return output.view(B,T,C), self.N*torch.sum(f_i * p_i)

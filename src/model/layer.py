import torch.nn as nn
if config.attention == "standard":
  from attention import multi_head
  
# Class for a single transformer layer/block
class layer(nn.Module):
  def __init__(self, C, T, nh):
    super().__init__()
    self.LN0 = nn.LayerNorm(C)
    if config.attention == "standard":
      self.attn = multi_head(C, T, nh)
    self.LN1 = nn.LayerNorm(C)
    self.MLP = nn.Sequential( nn.Linear(C, 4*C) , nn.GELU() , nn.Linear(4*C, C))
  def forward(self, E):
    B,T,C = E.shape
    # LayerNorm + Attention
    del_E = self.attn(self.LN0(E))
    # Residuals
    E_prime = E + del_E
    # LayerNorm & MLP + Residuals
    return E_prime + self.MLP(self.LN1(E_prime))

import torch.nn as nn

if config.attention == "standard":
  from attention import multi_head
if config.attention == "flash":
  from attention import flash

if config.MoE == "TRUE":
  if config.MoE_type == "switch":
    from MoE import switch_MoE_block
  if config.MoE_type == "top-k"
    from MoE import MoE_block
  else: print("Mixture of Experts type not recognized")
  
# Class for a single transformer layer/block
class layer(nn.Module):
  def __init__(self, C, T, nh, k=2, N=4):
    super().__init__()
    self.LN0 = nn.LayerNorm(C)
    # Implementing correct attention type
    if config.attention == "standard":
      self.attn = multi_head(C, T, nh)
    self.LN1 = nn.LayerNorm(C)
    # Implementing correct MLP layer type
    if config.MoE != "TRUE":
      self.MLP = nn.Sequential( nn.Linear(C, 4*C) , nn.GELU() , nn.Linear(4*C, C))
    elif config.MoE_type == "switch":
      pass
    elif config.MoE_type == "top-k":
      self.MoE_block = MoE_block(T, C, N, k)
    B,T,C = E.shape
    # LayerNorm + Attention
    del_E = self.attn(self.LN0(E))
    # Residuals
    E_prime = E + del_E
    # LayerNorm & MLP + Residuals
    return E_prime + self.MLP(self.LN1(E_prime))
    if config.MoE != "TRUE":
      return E_prime + self.MLP(self.LN1(E_prime))
    elif config.MoE_type == "switch":
      pass
    elif config.MoE_type == "top-k":
      MoE_out, loss = self.MoE_block(self.LN1(E_prime))
      return E_prime + MoE_out, loss

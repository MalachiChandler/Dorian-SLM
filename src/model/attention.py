import torch.nn as nn
from torch.nn import functional as F

# Class for multi-headed causal self attention block
class multi_head(nn.Module):
  def __init__(self, embedding_dim , T , nh):
    super().__init__()
    # making one big matrix for queries, keys, and values
    self.attn_block = nn.Linear(embedding_dim , 3*embedding_dim , bias=False)
    # perceptron layer for mixing information across heads at the end
    self.mixer = nn.Linear(embedding_dim , embedding_dim)
    # creating module mask for masking tokens > current timestep t
    self.register_buffer("mask", torch.tril(torch.ones(1, T, T)))
    self.nh = nh
  def forward(self, E):
    B,T,C = E.shape
    Q , K , V = self.attn_block(E).chunk(3,dim=-1)
    Q = Q.view(B , T , self.nh , C//self.nh).transpose(1,2)
    K = K.view(B , T , self.nh , C//self.nh).transpose(1,2)
    V = V.view(B , T , self.nh , C//self.nh).transpose(1,2)
    dot_table = Q @ K.transpose(-2,-1) / (K.shape[-1]**0.5)
    masked = dot_table.masked_fill(self.mask[:,:T,:T]==0 , float('-inf'))
    del_E = ((F.softmax(masked , -1)) @ V).transpose(1,2)
    return self.mixer(del_E.reshape(B,T,C))

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

# Creating class for model
class prototype(nn.Module):

  def __init__(self , V , C , T , n_heads=1):
    super().__init__()
    # creating embedding table that has a vector with C dimentions for each V token in the vocab
    self.embeddingTable = nn.Embedding(V , C)
    self.pos_embedding = nn.Embedding(T , C)
    # creating multi headed attention block
    self.attention = multi_head(C , T , n_heads)
    # Layer normalization
    self.LN = nn.LayerNorm(C)
    # Multilayer perceptron before logits
    self.MLP = nn.Sequential( nn.Linear(C, 4*C) , nn.ReLU() , nn.Linear(4*C, C))
    # creating single perceptron layer resposible for learning token predictions
    self.percep = nn.Linear(C , V)

  def forward(self, batch , target=None , nh=1):
    # Embedding
    E = self.embeddingTable(batch)
    B,T,C = E.shape
    E += self.pos_embedding(torch.arange(E.shape[1], device=E.device))
    # Multi-headed attention
    del_E = self.attention(E)
    # Residuals
    E_prime = E + del_E
    extra = self.LN(E_prime + self.MLP(E_prime))
    # Single layer output perceptron
    logits = self.percep(extra)

    if target != None:
      # calculating loss
      B,T,V = logits.shape
      loss = F.cross_entropy(logits.view(B*T,V) , target.view(B*T).long())
      return logits , loss
    else:
      return logits

  def generate(self, gen_len, device, seed_text=None):
    self.eval()
    seed_text = torch.zeros((1,1) , dtype=torch.long , device=device) if seed_text==None else seed_text.to(device)
    final_text = seed_text
    for i in range(gen_len):
      seed_text = seed_text[:,-T:]
      logits = self(seed_text)
      logits = logits[:,-1,:]
      pick = torch.multinomial(F.softmax(logits, -1) , 1)
      seed_text = torch.cat((seed_text , pick), dim=1)
      final_text = torch.cat((final_text , pick), dim=1)
    return final_text

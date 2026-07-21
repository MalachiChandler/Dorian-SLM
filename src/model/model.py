import torch.nn as nn
if config.layer == "standard":
  from layer import layer

# Creating class for model
class Dorian_SLM(nn.Module):

  def __init__(self , V , C , T , k=2, N=4, n_heads=1, n_layers=1):
    super().__init__()
    # creating embedding table that has a vector with C dimentions for each V token in the vocab
    self.embeddingTable = nn.Embedding(V , C)
    self.pos_embedding = nn.Embedding(T , C)
    # Transformer Layers including multiheaded attention and MLPs with LayerNormalization and residuals
    self.layers = nn.ModuleList([layer(C, T, n_heads, k, N) for i in range(n_layers)])
    self.LN = nn.LayerNorm(C)
    # creating single perceptron layer resposible for learning token predictions
    self.percep = nn.Linear(C , V, bias=False, )
    # Weight tying
    self.embeddingTable.weight = self.percep.weight
    # Initialization
    self.apply(self._init_weights)
    scale = 0.02 / math.sqrt(2 * n_layers)
    for block in self.layers:
        torch.nn.init.normal_(block.attn.mixer.weight, mean=0.0, std=scale)
        if config.MoE != "TRUE":
          torch.nn.init.normal_(block.MLP[2].weight, mean=0.0, std=scale)
        if config.MoE_type == "top-k":
          for expert in block.MoE_block.experts:
              torch.nn.init.normal_(expert.MLP[2].weight, mean=0.0, std=scale)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, batch , target=None , nh=1):
    # Embedding
    E = self.embeddingTable(batch)
    B,T,C = E.shape
    E += self.pos_embedding(torch.arange(E.shape[1], device=E.device))
    # Transformer Layers
    if config.MoE == "TRUE" and config.MoE_type == "top-k":
      MoE_loss = 0
      for layer in self.layers:
        E, MoE_loss_holder = layer(E)
        MoE_loss += MoE_loss_holder
    E = self.LN(E)
    # Single layer output perceptron
    logits = self.percep(E)

    if target != None:
      # calculating loss
      B,T,V = logits.shape
      loss = F.cross_entropy(logits.view(B*T,V) , target.view(B*T).long())
      if config.MoE == "TRUE" and config.MoE_type == "top-k":
        return logits , loss , MoE_loss/len(self.layers)
      else:
        return logits , loss
    else:
      return logits

  def generate(self, gen_len, device, seed_text=None, temperature=0.6):
    self.eval()
    seed_text = torch.zeros((1,1) , dtype=torch.long , device=device) if seed_text==None else seed_text.to(device)
    final_text = seed_text
    for i in range(gen_len):
      seed_text = seed_text[:,-T:]
      logits = self(seed_text)
      logits = logits[:,-1,:] / temperature
      pick = torch.multinomial(F.softmax(logits, -1) , 1)
      seed_text = torch.cat((seed_text , pick), dim=1)
      final_text = torch.cat((final_text , pick), dim=1)
    return final_text

  def show_params(self):
    param_count = sum(p.numel() for p in self.parameters())
    print(param_count)

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('../')  # Add the parent directory to the Python path

from Bigram import Block

# hyperparameters
batch_size = 64 # how many independent sequences to process in parallel?
block_size = 256 # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
 

torch.manual_seed(1337)


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", 'r', encoding='utf-8') as file:
    text = file.read()


# all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# mapping between chars and integers
stoi = { char: num for num, char in enumerate(chars)}
itos = { num: char for num, char in enumerate(chars)}
encode = lambda s: [stoi[char] for char in s] # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[num] for num in l]) # decoder: take a list of integers, output a string


# Train and Test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% Train, 10% Test
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (block_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class MahakabiBigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    
    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss


    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # crop idx to the last block_size token
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :]
            # softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # dim = -1 selects the last dimension i.e. C
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx


model = MahakabiBigramLanguageModel()
m = model.to(device)

#print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generate_text = decode(m.generate(context, max_new_tokens=2000)[0].tolist())

with open("generated_text.txt", "w") as file:
    file.write(generate_text)

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer('basic_english')
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def process(text):
    return torch.tensor(vocab(tokenizer(text)), dtype=torch.long)

train_iter = IMDB(split='train')
data = []
for label, text in train_iter:
    data.append((0 if label == "neg" else 1, process(text)))

input_size = len(vocab)
hidden_size = 64
output_size = 1
embedding_dim = 64
batch_size = 8

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        return self.sig(self.fc(out[-1]))

model = LSTMModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    total_loss = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        labels = torch.tensor([x[0] for x in batch], dtype=torch.float32).unsqueeze(1)
        texts = [x[1] for x in batch]
        texts = pad_sequence(texts).to(labels.device)

        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch Loss:", total_loss)

import torch
import torch.nn as nn
import torch.optim as optim

input_size = 5
hidden_size = 10
output_size = 1
seq_length = 4
batch_size = 2

x = torch.randn(seq_length, batch_size, input_size)
y = torch.randn(batch_size, output_size)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        last_out = out[-1]
        out = self.fc(last_out)
        return out

model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print("Output:", output)
print("Loss:", loss.item())

import torch
import tqdm

import random

import numpy as np

rec = np.load('test.npy')

x = torch.tensor(rec[:, 0], dtype=torch.float32)
y = torch.tensor(rec[:, 1], dtype=torch.float32)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lr1 = torch.nn.Linear(1, 200)
        self.lr2 = torch.nn.Linear(200, 200)
        self.lr3 = torch.nn.Linear(200, 200)
        self.lr4 = torch.nn.Linear(200, 1)

        self.par1 = torch.nn.Parameter(torch.rand(200), requires_grad=True)
        self.par2 = torch.nn.Parameter(torch.rand(200), requires_grad=True)

        self.prelu = torch.nn.PReLU()

    def forward(self, x): 
        b = x.shape[0]
        x = torch.div(x, 2000)

        x = self.lr1(x)
        x = self.prelu(x)

        x = self.lr2(x) + self.par1.unsqueeze(0).repeat(b, 1)
        # x = self.lr2(x)
        x = torch.sigmoid(x)

        x = self.lr3(x) + self.par2.unsqueeze(0).repeat(b, 1)
        # x = self.lr3(x)
        x = torch.sigmoid(x)

        
        x = self.lr4(x)
        return x

model = Net().train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
metric = torch.nn.MSELoss()

epochs = 100

avg_loss = 0
for e in tqdm.tqdm(range(epochs)):
    for xx, yy in zip(x, y):
        optimizer.zero_grad()

        inp = xx.unsqueeze(0).unsqueeze(0)
        outp = yy.unsqueeze(0).unsqueeze(0)
        
        res = model(inp)
        loss = metric(res, outp)
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()
    
print(avg_loss / (epochs * x.shape[0]))

model = model.eval()

i = random.randint(0, x.shape[0])
xx = x[i]
yy = y[i]

print(x[i], y[i])
print(xx, model(xx.unsqueeze(0).unsqueeze(0)), yy)
torch.save(model.state_dict(), 'test.pt')

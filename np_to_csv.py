import numpy as np

import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lr1 = torch.nn.Linear(1, 200)
        self.lr2 = torch.nn.Linear(200, 200)
        self.lr3 = torch.nn.Linear(200, 200)
        self.lr4 = torch.nn.Linear(200, 1)

        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        x = torch.div(x, 2000)

        x = self.lr1(x)
        x = self.prelu(x)

        x = self.lr2(x)
        x = torch.sigmoid(x)

        x = self.lr3(x)
        x = torch.sigmoid(x)
        
        x = self.lr4(x)
        return x

model = Net()
model.load_state_dict(torch.load('test.pt'))
model.eval()

rec = np.load('test.npy')

f = open('test.csv', 'w')
for r in rec:
    qq = torch.tensor(r[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    qq = model(qq)[0][0]
    f.write(f"{r[0]},{r[1]},{qq}\n")
f.close()
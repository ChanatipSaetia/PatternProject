import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from sklearn.base import BaseEstimator, TransformerMixin


class AutoEncoder(nn.Module, BaseEstimator, TransformerMixin):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_size),
            nn.Tanh())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=1e-5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, x, y=None):
        x = Variable(FloatTensor(x).cuda())
        for epoch in range(2000):
            # ===================forward=====================
            output = self(x)
            loss = self.criterion(output, x)
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ===================log========================
            if epoch % 100 == 99:
                print(loss.data.cpu().numpy()[0])
        return self

    def transform(self, x):
        x = Variable(FloatTensor(x).cuda())
        return self.encoder(x).data.cpu().numpy()

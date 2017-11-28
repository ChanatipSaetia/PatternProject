import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import Dataset, DataLoader


class DatasetForAutoEncoder(Dataset):

    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return self.datas.size()[0]

    def __getitem__(self, idx):
        return {
            'data': self.datas[idx],
            'label': self.labels[idx]
        }


class AutoEncoder(nn.Module, BaseEstimator, TransformerMixin):
    def __init__(self, input_size, output):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(input_size, output))
        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(output, input_size + 1),
            nn.Sigmoid())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.01)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, x, y=None, batch_size=200000, epochs=100):
        print(x.shape)
        print("AutoEncoder start training")

        self.train()
        x = FloatTensor(x)
        y = FloatTensor(y.astype(float))
        data = DatasetForAutoEncoder(x, y)
        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
        for epoch in range(epochs):
            sum_loss = 0
            sum_loss_pred = 0
            count = 0
            count_pred = 0
            for _, sample_batch in enumerate(dataloader):
                # ===================forward=====================
                self.optimizer.zero_grad()
                data = Variable(sample_batch['data'].cuda())
                label = Variable(sample_batch['label'].float().cuda())
                class_weight = label.size()[0] / (2 * torch.sum(label))
                output = self(data)
                loss = self.criterion(output[:, 0:-1], data)
                criterion_pred = nn.BCELoss()
                loss_pred = criterion_pred(output[:, -1], label)
                all_loss = loss + loss_pred
                # ===================backward====================
                all_loss.backward()
                self.optimizer.step()
                if epoch % 10 == 9:
                    sum_loss = sum_loss + loss.data.cpu().numpy()[0]
                    sum_loss_pred = sum_loss_pred + \
                        loss_pred.data.cpu().numpy()[0]
                    count = count + 1
                    count_pred = count_pred + 1
            # ===================log========================
            if epoch % 10 == 9:
                print(epoch, sum_loss / count, sum_loss_pred / count_pred)
        return self

    def transform(self, x):
        self.eval()
        x = Variable(FloatTensor(x).cuda(), volatile=True)
        return self.encoder(x).data.cpu().numpy()

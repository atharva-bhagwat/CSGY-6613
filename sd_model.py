import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class FCBlock(nn.Module):
    # Fully connected block
    def __init__(self):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, x):
      # forward pass for fc block
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      x = F.relu(x)
      x = self.fc4(x)
      x = F.relu(x)
      x = self.out(x)

      return F.log_softmax(x, dim=1) 

class BasicBlock(nn.Module):
    # Base class with train, test, pred, and save model methods
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.name = 'StateDescriptorRN'
    
    def train_(self, img, ques, ans):
        self.optimizer.zero_grad()
        output = self(img, ques)
        loss = F.nll_loss(output, ans)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(ans.data).cpu().sum()
        accuracy = correct * 100. / len(ans)
        return accuracy, loss
        
    def test_(self, img, ques, ans):
        output = self(img, ques)
        loss = F.nll_loss(output, ans)
        pred = output.data.max(1)[1]
        correct = pred.eq(ans.data).cpu().sum()
        accuracy = correct * 100. / len(ans)
        return accuracy, loss
        
    def save_model(self):
        torch.save(self.state_dict(), f"./model/{self.name}.pth")

class Dense(BasicBlock):
    # Fully connected block
    def __init__(self, batch_size=64):
        super(Dense, self).__init__()
      
        # g theta
        self.g_fc1 = nn.Linear(13, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        # # restructure image and question embeddings
        # self.coord_oi = torch.FloatTensor(batch_size, 2)
        # self.coord_oj = torch.FloatTensor(batch_size, 2)
        # self.coord_tensor = torch.FloatTensor(batch_size, 25, 2)
        
        # self.coord_oi = self.coord_oi.cuda()
        # self.coord_oj = self.coord_oj.cuda()
        # self.coord_tensor = self.coord_tensor.cuda()
        
        # self.coord_oi = Variable(self.coord_oi)
        # self.coord_oj = Variable(self.coord_oj)
        # self.coord_tensor = Variable(self.coord_tensor)
        
        # def cvt_coord(i):
        #     return [(i/5-2)/2., (i%5-2)/2.]
        
        # np_coord_tensor = np.zeros((batch_size, 25, 2))
        # for i in range(25):
        #     np_coord_tensor[:,i,:] = np.array(cvt_coord(i))
        
        # self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        # f phi
        self.fcout = FCBlock()
        
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, img, ques):
        # forward pass for RN
        print('img: ', img.shape)
        mb = img.size()[0] # 64
        n_channels = img.size()[1] # 1
        d = img.size()[2] # 6
        h = img.size()[3] # 4
        x_flat = img.view(mb, n_channels, d*h).permute(0,2,1) # 64, 1, 24 -> 64, 24, 1
        # x_flat = torch.cat([x_flat, self.coord_tensor], 2)
        print('x_flat ',x_flat.shape)
        
        ques = torch.unsqueeze(ques, 1)
        ques = ques.repeat(1, d*h, 1)
        ques = torch.unsqueeze(ques, 2)
        print('ques: ',ques.shape)
        
        x_i = torch.unsqueeze(x_flat, 1)
        x_i = x_i.repeat(1, d*h, 1, 1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = torch.cat([x_j, ques], 3)
        x_j = x_j.repeat(1, 1, d*h, 1)

        print('x_i: ',x_i.shape)
        print('x_j: ',x_j.shape)
        
        x_full = torch.cat([x_i, x_j], 3)
        print('x_full: ',x_full.shape)
        x_ = x_full.view(mb * (d*h) * (d*h), 13)
        
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        x_g = x_.view(mb, (d*h) * (d*h), 256)
        x_g = x_g.sum(1).squeeze()
        
        return self.fcout(x_g)
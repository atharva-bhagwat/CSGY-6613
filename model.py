import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batch_norm1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batch_norm4(x)
        
        return x
        
class FCBlock(nn.Module):
    def __init__(self):
        super(FCBlock, self).__init__()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
        
class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.name = 'RN'
    
    def train_(self, img, ques, label):
        self.optimizer.zero_grad()
        output = self(img, ques)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, img, ques, label):
        output = self(img, ques)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def pred_(self, img, ques):
        output = self(img, ques)
        return output.data.max(1)[1]
        
    def save_model(self):
        torch.save(self.state_dict(), f"./model/{self.name}.pth")
        
class RN(BasicBlock):
    def __init__(self, batch_size=64):
        super(RN, self).__init__()
        
        self.conv = ConvBlock()
        
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)
        
        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = Variable(torch.FloatTensor(batch_size, 2))
        self.coord_oj = Variable(torch.FloatTensor(batch_size, 2))
        self.coord_tensor = Variable(torch.FloatTensor(batch_size, 25, 2))
        
        self.coord_oi = self.coord_oi.cuda()
        self.coord_oj = self.coord_oj.cuda()
        self.coord_tensor = self.coord_tensor.cuda()
        
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)
        self.coord_tensor = Variable(self.coord_tensor)
        
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        np_coord_tensor = np.zeros((batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array(cvt_coord(i))
        
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        self.fcout = FCBlock()
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, img, ques):
        x = self.conv(img)
        # 64(batch) * 24 * 5 * 5
        # print("x shape ",x.shape)
        mb = x.size()[0] # 64
        n_channels = x.size()[1] # 24
        d = x.size()[2] # 5
        x_flat = x.view(mb, n_channels, d*d).permute(0,2,1)
        # x.view(mb, n_channels, d*d) -> (64, 24, 25)
        # after permute -> (64, 25, 24)
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)
        
        ques = torch.unsqueeze(ques, 1)
        ques = ques.repeat(1, 25, 1)
        ques = torch.unsqueeze(ques, 2)
        
        x_i = torch.unsqueeze(x_flat, 1)
        x_i = x_i.repeat(1, 25, 1, 1)
        x_j = torch.unsqueeze(x_flat, 2)
        x_j = torch.cat([x_j, ques], 3)
        x_j = x_j.repeat(1, 1, 25, 1)
        
        x_full = torch.cat([x_i, x_j], 3)
        # print("x_full ", x_full.view(mb * (d*d) * (d*d), 63))
        x_ = x_full.view(mb * (d*d) * (d*d), 63)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        x_g = x_.view(mb, (d*d) * (d*d), 256)
        x_g = x_g.sum(1).squeeze()
        
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_g)
        
        return self.fcout(x_f)
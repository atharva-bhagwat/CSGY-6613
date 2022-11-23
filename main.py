import os
import gc
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import RN
from util import translate

gc.collect()
torch.cuda.empty_cache()

BATCH_SIZE = 64
EPOCHS = 20

model = RN()

input_img = torch.FloatTensor(BATCH_SIZE, 3, 75, 75)
input_qst = torch.FloatTensor(BATCH_SIZE, 11)
input_ans = torch.LongTensor(BATCH_SIZE)

model.cuda()
input_img = input_img.cuda()
input_qst = input_qst.cuda()
input_ans = input_ans.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
input_ans = Variable(input_ans)
        
def setup_dir():
    os.makedirs("model", exist_ok=True)
    
def load_data(folder="sort_of_clevr", filename="sort_of_clevr.pkl"):
    path = os.path.join(folder, filename)
    with open(path, "rb") as fd:
        train, test = pickle.load(fd)
        
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    
    print('Processing data...')
    
    for img, rel, norel in train:
        img = np.swapaxes(img, 0, 2)
        
        for qst, ans in zip(rel[0], rel[1]):
            rel_train.append((img, qst, ans))
        for qst, ans in zip(norel[0], norel[1]):
            norel_train.append((img, qst, ans))
    
    for img, rel, norel in test:
        img = np.swapaxes(img, 0, 2)
        
        for qst, ans in zip(rel[0], rel[1]):
            rel_test.append((img, qst, ans))
        for qst, ans in zip(norel[0], norel[1]):
            norel_test.append((img, qst, ans))
            
    return rel_train, rel_test, norel_train, norel_test
    
def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    input_ans.data.resize_(ans.size()).copy_(ans)
    
def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)
    
def train(rel_data, norel_data):
    model.train()
    
    random.shuffle(rel_data)
    random.shuffle(norel_data)
    
    rel_data = cvt_data_axis(rel_data)
    norel_data = cvt_data_axis(norel_data)
    
    acc_rel = 0.0
    acc_norel = 0.0
    
    loss_rel = 0.0
    loss_norel = 0.0
    
    for batch_idx in range(len(rel_data[0]) // BATCH_SIZE):
        tensor_data(rel_data, batch_idx)
        acc, loss = model.train_(input_img, input_qst, input_ans)
        
        acc_rel += acc
        loss_rel += loss
        
        tensor_data(norel_data, batch_idx)
        acc, loss = model.train_(input_img, input_qst, input_ans)
        
        acc_norel += acc
        loss_norel += loss
        
    acc_rel /= len(rel_data[0]) // BATCH_SIZE
    loss_rel /= len(rel_data[0]) // BATCH_SIZE
    
    acc_norel /= len(rel_data[0]) // BATCH_SIZE
    loss_norel /= len(rel_data[0]) // BATCH_SIZE
    
    return acc_rel, acc_norel, loss_rel, loss_norel
    
def test(rel_data, norel_data):
    model.eval()
    
    rel_data = cvt_data_axis(rel_data)
    norel_data = cvt_data_axis(norel_data)
    
    acc_rel = 0.0
    acc_norel = 0.0
    
    loss_rel = 0.0
    loss_norel = 0.0
    
    for batch_idx in range(len(rel_data[0]) // BATCH_SIZE):
        tensor_data(rel_data, batch_idx)
        acc, loss = model.test_(input_img, input_qst, input_ans)
        
        acc_rel += acc
        loss_rel += loss
        
        tensor_data(norel_data, batch_idx)
        acc, loss = model.test_(input_img, input_qst, input_ans)
        
        acc_norel += acc
        loss_norel += loss
        
    acc_rel /= len(rel_data[0])
    loss_rel /= len(rel_data[0])
    
    acc_norel /= len(rel_data[0])
    loss_norel /= len(rel_data[0])
    
    return acc_rel, acc_norel, loss_rel, loss_norel

def predict_one(rel_data, norel_data):
    model.eval()
    
    rel_data = cvt_data_axis(rel_data)
    norel_data = cvt_data_axis(norel_data)
    
    tensor_data(rel_data, 0)
    pred_rel = model.pred_(input_img, input_qst)
    
    tensor_data(norel_data, 0)
    pred_norel = model.pred_(input_img, input_qst)
    
    return pred_rel[0].cpu().numpy(), pred_norel[0].cpu().numpy()

def driver():
    setup_dir()
    
    rel_train, rel_test, norel_train, norel_test = load_data()
    
    train_acc_rel = []
    train_loss_rel = []
    
    train_acc_norel = []
    train_loss_norel = []
    
    test_acc_rel = []
    test_loss_rel = []
    
    test_acc_norel = []
    test_loss_norel = []
    
    print('Training...')
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}/{EPOCHS}')
        acc_rel, acc_norel, loss_rel, loss_norel = train(rel_train, norel_train)
        
        train_acc_rel.append(acc_rel)
        train_acc_norel.append(acc_norel)
        train_loss_rel.append(loss_rel)
        train_loss_norel.append(loss_norel)
        
        print(f'Train Set:\n\tRelational Acc: {acc_rel}\tRelational Loss: {loss_rel}\n\tNon-Relational Acc: {acc_norel}\tNon-Relational Loss: {loss_norel}')
        
        acc_rel, acc_norel, loss_rel, loss_norel = test(rel_test, norel_test)
        
        test_acc_rel.append(acc_rel)
        test_acc_norel.append(acc_norel)
        test_loss_rel.append(loss_rel)
        test_loss_norel.append(loss_norel)
        
        print(f'Test Set:\n\tRelational Acc: {acc_rel}\tRelational Loss: {loss_rel}\n\tNon-Relational Acc: {acc_norel}\tNon-Relational Loss: {loss_norel}')
    
    model.save_model()
    
    img, rel_qst, rel_ans = rel_test[0]
    _, norel_qst, norel_ans = norel_test[0]
    
    rel_pred, norel_pred = predict_one(rel_test, norel_test)
    
    data_entry = (img, (rel_qst, rel_ans, rel_pred), (norel_qst, norel_ans, norel_pred))
    
    translate(data_entry)
    
if __name__ == "__main__":
    driver()
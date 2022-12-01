import os
import random
import pickle
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import RN
from util import translate

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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

os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)
    
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
        
        acc_rel += acc.item()
        loss_rel += loss.item()
        
        tensor_data(norel_data, batch_idx)
        acc, loss = model.train_(input_img, input_qst, input_ans)
        
        acc_norel += acc.item()
        loss_norel += loss.item()
        
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
        
        acc_rel += acc.item()
        loss_rel += loss.item()
        
        tensor_data(norel_data, batch_idx)
        acc, loss = model.test_(input_img, input_qst, input_ans)
        
        acc_norel += acc.item()
        loss_norel += loss.item()
        
    acc_rel /= len(rel_data[0]) // BATCH_SIZE
    loss_rel /= len(rel_data[0]) // BATCH_SIZE
    
    acc_norel /= len(rel_data[0]) // BATCH_SIZE
    loss_norel /= len(rel_data[0]) // BATCH_SIZE
    
    return acc_rel, acc_norel, loss_rel, loss_norel
    
def generate_plot(train_data_rel, train_data_norel, test_data_rel, test_data_norel, label):
    plt.figure()
    plt.plot(range(EPOCHS), train_data_rel, '-', label=f"train_{label}_rel")
    plt.plot(range(EPOCHS), train_data_norel, '-', label=f"train_{label}_norel")
    plt.plot(range(EPOCHS), test_data_rel, '-', label=f"test_{label}_rel")
    plt.plot(range(EPOCHS), test_data_norel, '-', label=f"test_{label}_norel")
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("output", f"{label}.jpg"))
    print(f'{os.path.join("output", f"{label}.jpg")} saved...')
    
def test_single(rel_test, norel_test, idx):
    img, rel_qst, rel_ans = rel_test[idx]
    _, norel_qst, norel_ans = norel_test[idx]
    
    rel_pred, norel_pred = predict_one(rel_test, norel_test, idx=idx)
    
    data_entry = (img, (rel_qst, rel_ans, rel_pred), (norel_qst, norel_ans, norel_pred))
    
    translate(data_entry, filename=f'test_{idx}.jpg')

def predict_one(rel_data, norel_data, idx):
    model.eval()
    
    rel_data = cvt_data_axis(rel_data)
    norel_data = cvt_data_axis(norel_data)
    
    tensor_data(rel_data, 0)
    pred_rel = model.pred_(input_img, input_qst)
    
    tensor_data(norel_data, 0)
    pred_norel = model.pred_(input_img, input_qst)
    
    return pred_rel[idx].cpu().numpy(), pred_norel[idx].cpu().numpy()

def driver(mode):
    rel_train, rel_test, norel_train, norel_test = load_data()
    if mode == "train":
        print(f'Training mode...')
        train_acc_rel = []
        train_loss_rel = []
        
        train_acc_norel = []
        train_loss_norel = []
        
        test_acc_rel = []
        test_loss_rel = []
        
        test_acc_norel = []
        test_loss_norel = []
        
        for epoch in range(EPOCHS):
            print(f'Epoch: {epoch+1}/{EPOCHS}')
            acc_rel, acc_norel, loss_rel, loss_norel = train(rel_train, norel_train)
            
            train_acc_rel.append(acc_rel)
            train_acc_norel.append(acc_norel)
            train_loss_rel.append(loss_rel)
            train_loss_norel.append(loss_norel)
            
            print(f'Train Set:\n\tRelational Acc: {acc_rel:.2f}\tRelational Loss: {loss_rel:.3f}\n\tNon-Relational Acc: {acc_norel:.2f}\tNon-Relational Loss: {loss_norel:.3f}')
            
            acc_rel, acc_norel, loss_rel, loss_norel = test(rel_test, norel_test)
            
            test_acc_rel.append(acc_rel)
            test_acc_norel.append(acc_norel)
            test_loss_rel.append(loss_rel)
            test_loss_norel.append(loss_norel)
            
            print(f'Test Set:\n\tRelational Acc: {acc_rel:.2f}\tRelational Loss: {loss_rel:.3f}\n\tNon-Relational Acc: {acc_norel:.2f}\tNon-Relational Loss: {loss_norel:.3f}')
            
        # generate acc plt
        generate_plot(train_acc_rel, train_acc_norel, test_acc_rel, test_acc_norel, "acc")
        
        # generate loss plt
        generate_plot(train_loss_rel, train_loss_norel, test_loss_rel, test_loss_norel, "loss")
        
        model.save_model()
    else:
        print(f'Testing mode...')
        model.load_state_dict(torch.load(os.path.join("model","RN.pth")))
        
    
    # test id 0 
    test_single(rel_test, norel_test, 0)
    
    # test id 10 
    test_single(rel_test, norel_test, 15)
    
    # test id 20 
    test_single(rel_test, norel_test, 30)
    
    # test id 30 
    test_single(rel_test, norel_test, 45)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set mode: Train/Test")
    parser.add_argument("--mode", type=str, choices=["train","test"], default="train")
    args = parser.parse_args()
    driver(mode=args.mode)
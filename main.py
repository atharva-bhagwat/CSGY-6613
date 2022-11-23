"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import os
import pickle
import random
import numpy as np

import torch
from torch.autograd import Variable

from model import RN
from util import translate

model = RN()
relation_type='binary'
epochs=20
model_dirs = './model'
bs = 64
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 11)
label = torch.LongTensor(bs)

if torch.cuda.is_available():
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    

    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_rels = []
    acc_norels = []

    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())


    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    return avg_acc_binary, avg_acc_unary, avg_loss_binary, avg_loss_unary

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []

    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)

    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    return accuracy_rel, accuracy_norel, loss_binary, loss_unary

    
def load_data():
    print('loading data...')
    dirs = './sort_of_clevr/'
    filename = os.path.join(dirs,'sort_of_clevr.pkl')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)

    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:

        img = np.swapaxes(img, 0, 2)

        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)

        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)
    
def predict(rel_train, rel_test, norel_train, norel_test):
  dataset=[]

  try:
    os.makedirs(model_dirs)
  except:
    print('directory {} already exists'.format(model_dirs))

  #print(f"Training {model} {f'({relation_type})'}")

  for epoch in range(1,1+1):
    print("Epoch: ",epoch,"/",epochs)
    train_acc_binary, train_acc_unary,train_loss_binary, train_loss_unary = train(
        epoch, rel_train, norel_train)
    print('\nTrain set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}% | Binary loss: {:.0f}% | Unary loss: {:.0f}%'
    .format(train_acc_binary, train_acc_unary,train_loss_binary, train_loss_unary))
    test_acc_binary, test_acc_unary, test_loss_binary, train_loss_unary = test(
        epoch,rel_test, norel_test)
    print('Test set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}% | Binary loss: {:.0f}% | Unary loss: {:.0f}%\n'
    .format(test_acc_binary, test_acc_unary, test_loss_binary, train_loss_unary))

rel_train, rel_test, norel_train, norel_test = load_data()
predict(rel_train, rel_test, norel_train, norel_test)

for i in range(len(rel_train)):
  image,rel_ques,rel_ans=rel_train[i]
  _,norel_ques,norel_ans=norel_train[i]
  # print(rel_ques,rel_ans)
  # print("------------")
  # print(norel_ques,norel_ans)
  translate([image,(rel_ques,rel_ans),(norel_ques,norel_ans)])

model.save_model()

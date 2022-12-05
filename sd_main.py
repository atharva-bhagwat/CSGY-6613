import os
import random
import pickle
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sd_model import Dense

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# initialize global variables
BATCH_SIZE = 64
EPOCHS = 25

# create model object
model = Dense()

# create tensors for image, question, and answer for batching
input_img = torch.FloatTensor(BATCH_SIZE, 1, 4, 6)
input_qst = torch.FloatTensor(BATCH_SIZE, 11)
input_ans = torch.LongTensor(BATCH_SIZE)

# load model and tensors to GPU
model.cuda()
input_img = input_img.cuda()
input_qst = input_qst.cuda()
input_ans = input_ans.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
input_ans = Variable(input_ans)

# setup directories
os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

def tensor_data(data, i):
    """Helper function to generate batches

    Args:
        data (list): Training or testing set
        i (int): Index of current pointer
    """
    img = torch.from_numpy(np.asarray(data[0][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][BATCH_SIZE*i:BATCH_SIZE*(i+1)]))

    # copy batches into tensors
    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    input_ans.data.resize_(ans.size()).copy_(ans)
    
def cvt_data_axis(data):
    """Helper function to restructure data

    Args:
        data (int): Training or testing set

    Returns:
        tuple: tuple of lists: image, question, answer
    """
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

def load_data(folder="sort_of_clevr", filename="sort_of_clevr_descriptor.pkl"):
    """Helper function to load and restructure dataset into lists

    Args:
        folder (str): Path to dataset. Defaults to "sort_of_clevr".
        filename (str): Pickle filename. Defaults to "sort_of_clevr_descriptor.pkl".

    Returns:
        list, list, list, list: relational training data, relational testing data, 
                                non-relational training data, non-relational testing
    """
    path = os.path.join(folder, filename)
    with open(path, "rb") as fd:
        train, test = pickle.load(fd)
        
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    
    print('Processing data...')
    
    # data is restructed as : [image, question, answer] for all images in training and testing set
    for img, rel, norel in train:
      img = np.swapaxes(img, 0, 2)    # swap 1st and 3rd axis
      for qst, ans in zip(rel[0], rel[1]):
          rel_train.append((img, qst, ans))
      for qst, ans in zip(norel[0], norel[1]):
          norel_train.append((img, qst, ans))
    
    for img, rel, norel in test:
      img = np.swapaxes(img, 0, 2)    # swap 1st and 3rd axis
      for qst, ans in zip(rel[0], rel[1]):
          rel_test.append((img, qst, ans))
      for qst, ans in zip(norel[0], norel[1]):
          norel_test.append((img, qst, ans))
    
    return rel_train, rel_test, norel_train, norel_test
    
def train(rel_data, norel_data):
    """Train method

    Args:
        rel_data (list): Training relational data
        norel_data (list): Training non-relational data

    Returns:
        float, float, float, float: relational accuracy, non-relational accuracy,
                                    relational loss, non-relational loss
    """
    model.train()
    
    # shuffle data for training
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
        acc, loss = model.train_(input_img, input_qst, input_ans)   # pass to model
        
        acc_norel += acc.item()
        loss_norel += loss.item()
        
    acc_rel /= len(rel_data[0]) // BATCH_SIZE
    loss_rel /= len(rel_data[0]) // BATCH_SIZE
    
    acc_norel /= len(rel_data[0]) // BATCH_SIZE
    loss_norel /= len(rel_data[0]) // BATCH_SIZE
    
    return acc_rel, acc_norel, loss_rel, loss_norel
    
def test(rel_data, norel_data):
    """Test method

    Args:
        rel_data (list): Testing relational data
        norel_data (list): Testing non-relational data

    Returns:
        float, float, float, float: relational accuracy, non-relational accuracy,
                                    relational loss, non-relational loss
    """
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
        acc, loss = model.test_(input_img, input_qst, input_ans)    # pass to model
        
        acc_norel += acc.item()
        loss_norel += loss.item()
        
    acc_rel /= len(rel_data[0]) // BATCH_SIZE
    loss_rel /= len(rel_data[0]) // BATCH_SIZE
    
    acc_norel /= len(rel_data[0]) // BATCH_SIZE
    loss_norel /= len(rel_data[0]) // BATCH_SIZE
    
    return acc_rel, acc_norel, loss_rel, loss_norel
    
def generate_plot(train_data_rel, train_data_norel, test_data_rel, test_data_norel, label):
    """Helper function to generate plots

    Args:
        train_data_rel (list): Relational training accuracy/loss
        train_data_norel (list): Non-relational training accuracy/loss
        test_data_rel (list): Relational test accuracy/loss
        test_data_norel (list): Non-relational test accuracy/loss
        label (str): Label for title
    """
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

def driver(mode):
    """Main function with training and testing loop

    Args:
        mode (str): train/test, if mode is train, train a model else just load model from saved path
    """
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
        generate_plot(train_acc_rel, train_acc_norel, test_acc_rel, test_acc_norel, "acc_sd")
        
        # generate loss plt
        generate_plot(train_loss_rel, train_loss_norel, test_loss_rel, test_loss_norel, "loss_sd")
        
        model.save_model()
    else:
        print(f'Testing mode...')
        model.load_state_dict(torch.load(os.path.join("model","StateDescriptorRN.pth")))
        
    acc_rel, acc_norel, loss_rel, loss_norel = test(rel_train, norel_train)
    print('Final metrics on train set:')
    print(f'Relational Accuracy: {acc_rel:.2f}\tNon-relational Accuracy: {acc_norel:.2f}')
    print(f'Relational Loss: {loss_rel:.3f}\tNon-relational Loss: {loss_norel:.3f}')
    acc_rel, acc_norel, loss_rel, loss_norel = test(rel_test, norel_test)
    print('Final metrics on test set:')
    print(f'Relational Accuracy: {acc_rel:.2f}\tNon-relational Accuracy: {acc_norel:.2f}')
    print(f'Relational Loss: {loss_rel:.3f}\tNon-relational Loss: {loss_norel:.3f}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set mode: Train/Test")
    parser.add_argument("--mode", type=str, choices=["train","test"], default="train")
    args = parser.parse_args()
    driver(mode=args.mode)
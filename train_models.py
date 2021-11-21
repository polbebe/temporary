import torch
from torch import nn
import numpy as np
from collections import deque
from models.bnn import BNN
import os
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
from torch.multiprocessing import Process, Queue

def robotTrain(env, robot, Net, data_path, model_path):
    data = np.load(data_path + env + '_train_1000_' + str(robot) + '.npy', allow_pickle=True)
    trainer = ModelTrainer(data, Net)
    model, L, Ls = trainer.train()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path + env + '_model_' + str(robot) + '.pt')

def robotReTrain(data, model):
    t = ModelTrainer(data, BNN)
    t.set_model(model)
    model, L, Ls = t.train()
    return model, L, Ls

def robotPreTrain(n, neighbors, Net, data_path, model_path):
    env = neighbors[0][1].split("_")[0]
    robot = neighbors[0][1].split("_")[1]
    data = np.load(data_path + env + '_train_1000_' + str(robot) + '.npy', allow_pickle=True)
    for i in range(1, len(neighbors)):
        env = neighbors[i][1].split("_")[0]
        robot = neighbors[i][1].split("_")[1]

        curr_data = np.load(data_path + env + '_train_1000_' + str(robot) + '.npy', allow_pickle=True)

        data = np.concatenate((data, curr_data), axis=0)

    trainer = ModelTrainer(data, Net)
    model, L, Ls = trainer.train()
    # plt.plot(np.arange(len(Ls)), Ls)

    env = n.split("_")[0]
    robot = n.split("_")[1]
    fresh_data = np.load(data_path + env + '_train_1000_' + str(robot) + '.npy', allow_pickle=True)
    model, L, Ls = robotReTrain(fresh_data, model)
    # plt.plot(np.arange(len(Ls)), Ls)
    # plt.grid()
    # plt.show()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path + env + '_model_' + str(robot) + '.pt')
    return L

# Multiprocessing function to be run
def run_process(q_in, q_out, Net):
    id = q_in.get()
    print('Process '+str(id)+' started.')
    while True:
        obj = q_in.get()
        if obj is None:
            break
        env, i = obj
        data = np.load('data/' + env + '_train_10000_' + str(i) + '.npy', allow_pickle=True)
        trainer = ModelTrainer(data, Net)
        model, L, Ls = trainer.train()
        if not os.path.exists('trained/'):
            os.makedirs('trained/')
        torch.save(model, 'trained/' + env + '_model_' + str(i) + '.pt')
        print('Model ' + env + ' ' + str(i) + ' trained with final Loss: ' + str(L))
        q_out.put(id)
    print('Process '+str(id)+' ended.')

#  Function for converting data from transition format into a more usable model training format
def prep_data(data, sequential=False):
    d = np.float32
    X, A, Y = [], [], []
    seq_len = 10
    x, a, y = deque(maxlen=seq_len), deque(maxlen=seq_len), deque(maxlen=seq_len)
    for step in data:
        if not step[-2]:
            if sequential and len(x) == seq_len:
                X.append(np.array(x))
                A.append(np.array(a))
                Y.append(np.array(y))
            else:
                X.append(step[0])
                A.append(step[1])
                Y.append(step[3])
            x.append(step[0])
            a.append(step[1])
            y.append(step[3])
        if step[-2] and sequential:
            x, a, y = deque(maxlen=seq_len), deque(maxlen=seq_len), deque(maxlen=seq_len)
    return np.array(X, dtype=d), np.array(A, dtype=d), np.array(Y, dtype=d)

# shuffles and puts data into batches
def shuffle_batch(data, batch_size=256):
    X, A, Y = data
    p = np.random.permutation(len(X))
    X, A, Y = X[p], A[p], Y[p]
    batches = []
    while len(X) > 0:
        x, a, y = X[:batch_size], A[:batch_size], Y[:batch_size]
        X, A, Y = X[batch_size:], A[batch_size:], Y[batch_size:]
        batches.append((x, a, y))
    return batches

# Trainer class for our models, it is initialized and used in parallel
class ModelTrainer:
    def __init__(self, data, Net, hid_size=None, lr=1e-3):
        self.x_dim = data[0][0].shape[0]
        self.a_dim = data[0][1].shape[0]
        self.y_dim = data[0][3].shape[0]
        self.lr = lr
        if hid_size is not None:
            self.model = Net(self.x_dim, self.a_dim, hid_size).to(device)
        else:
            self.model = Net(self.x_dim, self.a_dim).to(device)
        # self.data = prep_data(data)
        self.data = data

    def set_model(self, model):
        self.model = model

    def validate(self, data):
        L = 0.0
        data = prep_data(data)
        batches = shuffle_batch(data)
        for batch in batches:
            x, a, y = batch
            x, a, y = torch.from_numpy(x).to(device), torch.from_numpy(a).to(device), torch.from_numpy(y).to(device)
            pred = self.model(x, a, train=True)
            l = self.model.loss(pred, y)
            L += l.item() * x.shape[0]
        L = L / len(data[0])
        return L

    def train(self, epochs=1000, verbose=0, batch_size=64):
        Ls = []
        # data = [(step[0], step[1], step[3]) for step in self.data]
        data = self.data[:,(0,1,3)]
        for epoch in range(epochs):
            states, actions, targets = self.model.sample(data, batch_size=64)
            loss = self.model.train_step(states, actions, targets)
            Ls.append(loss)
        return self.model, Ls[-1], Ls

    # def train(self, epochs=100, verbose=0):
    #     L = 0.0
    #     Ls = []
    #     for epoch in range(epochs):
    #         batches = shuffle_batch(self.data)
    #         self.optim.zero_grad()
    #         L = 0.0
    #         for batch in batches:
    #             x, a, y = batch
    #             x, a, y = torch.from_numpy(x).to(device), torch.from_numpy(a).to(device), torch.from_numpy(y).to(device)
    #             pred = self.model(x, a, train=True)
    #             l = self.model.loss(pred, y)
    #             l.backward()
    #             L += l.item()*x.shape[0]
    #         self.optim.step()
    #         L = L / len(self.data[0])
    #         Ls.append(L)
    #         if verbose > 0 and epoch%verbose == 0:
    #             print('Epoch: '+str(epoch)+' Avg Loss: '+str(L))
    #     return self.model, L, Ls

def train_all():
    envs = ['Ant', 'Crawler', 'Dog', 'Spindra']
    tests = [(env, i) for env in envs for i in range(100)]
    n_cpu = 8
    Qs = []
    Ps = []
    busy = []
    for i in range(n_cpu):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=run_process, args=(q_in,q_out, BNN))
        p.start()
        q_in.put(i)
        Qs.append((q_in,q_out))
        Ps.append(p)
        busy.append(False)
    idx = 0
    while idx < len(tests):
        for i in range(len(busy)):
            if busy[i] == False:
                Qs[i][0].put(tests[idx])
                busy[i] = True
                idx += 1
            else:
                try:
                    if Qs[i][1].get_nowait() == i:
                        busy[i] = False
                except: pass

    for i in range(len(busy)):
        Qs[i][0].put(None)
        Qs[i][1].get()
    print('Done')

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        pass
    train_all()
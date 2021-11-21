import torch
import numpy as np
from torch import nn, optim

class Linear(nn.Module):
    def __init__(self, state_dim, act_dim, n_hid=200, lr=1e-4):
        super(Linear, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim+act_dim, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, n_hid),
                                              nn.ReLU(),
                                              nn.Linear(n_hid, state_dim)
                                            )
        self.softplus = nn.Softplus()
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_mean = None
        self.to(self.device)
        self.n_steps = 10
        self.teacher_forcing = False

    def set_norm(self, mean,std):
        self.norm_mean = mean.to(self.device).float()
        self.norm_std = std.to(self.device).float()
        self.norm_std[self.norm_std < 1e-12] = 1.0

    def forward(self, states, actions, train=False):
        if not torch.is_tensor(states):
            states = torch.from_numpy(states).to(self.device).to(torch.float32)
        if not torch.is_tensor(actions):
            actions = torch.from_numpy(actions).to(self.device).to(torch.float32)
        x = torch.cat([states, actions], dim=-1)

        # Input Normalization
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std

        mean = self.model(x)
        mean = mean + states
        return mean

    def loss(self, pred, target):
        l = torch.mean((pred - target)**2)
        return l

    def get_loss(self, states, actions, targets):
        preds = self.forward(states, actions, train=True)
        loss = self.loss(preds, targets)
        return loss

    def train_step(self, states, actions, targets):
        self.model.train()
        self.optimizer.zero_grad()
        L = 0.0
        s = states[:, 0, :]
        for i in range(self.n_steps):
            if self.teacher_forcing == True:
                s = states[:, i, :]
            preds = self.forward(s, actions[:,i,:], train=True)
            l = self.loss(preds, targets[:,i,:])
            if self.teacher_forcing == False:
                s = preds
            L += l
        loss = L / self.n_steps
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def sample(self, data, batch_size=256):
        P = np.random.permutation(len(data)-self.n_steps)[:batch_size]
        states, actions, targets = [], [], []
        for p in P:
            S, A, T = [], [], []
            for i in range(self.n_steps):
                s, a, t = data[p]
                S.append(s)
                A.append(a)
                T.append(t)
            states.append(np.array(S))
            actions.append(np.array(A))
            targets.append(np.array(T))
        states, actions, targets = np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(targets, dtype=np.float32)
        states, actions, targets = torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(targets)
        states, actions, targets = states.to(self.device), actions.to(self.device), targets.to(self.device)
        return states, actions, targets
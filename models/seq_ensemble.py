from torch import nn, optim
import torch
import numpy as np

class Ensemble(nn.Module):
    # def __init__(self, models, n_elites=-1):
    #     super(Ensemble, self).__init__()
    #     if n_elites > 0:
    #         self.n_elites = n_elites
    #     else:
    #         self.n_elites = len(models)
    #     self.pop_size = len(models)
    #     self.models = models
    #     self.elite_counter = np.zeros(self.pop_size)
    #     self.elite_idx = np.random.permutation(self.pop_size)[:self.n_elites]
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, models, pop_size, n_elites):
        super(Ensemble, self).__init__()
        self.pop_size = pop_size
        self.n_elites = n_elites
        self.models = models
        self.elite_counter = np.zeros(self.pop_size)
        self.elite_idx = np.random.permutation(self.pop_size)[:self.n_elites]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        for i in range(self.pop_size):
            self.models[i].to(device)
        return self

    def set_lr(self, lr):
        for model in self.models:
            model.lr = lr
            model.optimizer = optim.Adam(model.parameters(), lr=model.lr)

    def set_norm(self, mean,std):
        for i in range(self.pop_size):
            self.models[i].set_norm(mean, std)

    def forward(self, states, actions, train=False):
        # preds = [self.models[i](states, actions) for i in self.elite_idx]
        i = self.elite_idx[np.random.randint(0, self.n_elites)]
        # i = np.random.randint(0, self.pop_size)
        pred = self.models[i](states, actions)
        return pred

    def loss(self, preds, targets):
        loss = []
        for model in self.models:
            loss.append(
                model.loss(preds, targets).item()
            )
        return np.mean(loss)

    def train_set(self, states, actions, targets):
        losses = []
        for model in self.models:
            loss = model.train_set(states, actions, targets)
            losses.append(loss)
        return np.mean(losses)

    def train_step(self, states, actions, targets):
        assert len(states) == len(actions) == len(targets)
        valid_losses = []
        train_losses = []
        for model in self.models:
            shuffle = np.random.permutation(len(states))
            train_len = int(0.875*len(states))
            train_idx = shuffle[:train_len]
            valid_idx = shuffle[train_len:]

            # This isn't a strict validation as what is chosen as validation here may be train another time
            # but it creates more diversity in the models by choosing only a subset and gives a little better validation
            train_states, train_actions, train_targets = states[train_idx], actions[train_idx], targets[train_idx]
            valid_states, valid_actions, valid_targets = states[valid_idx], actions[valid_idx], targets[valid_idx]
            train_loss = model.train_step(train_states, train_actions, train_targets)
            valid_loss = model.get_loss(valid_states, valid_actions, valid_targets)

            valid_losses.append(valid_loss.item())
            train_losses.append(train_loss)

        self.elite_idx = np.argsort(valid_losses)[:self.n_elites]

        return np.mean(train_losses)

    def validation_loss(self, states, actions, targets):
        # Caluculate the mean loss on the given validation set.
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        targets = torch.from_numpy(targets).to(self.device)
        return self.get_loss(states, actions, targets)

    def get_loss(self, states, actions, targets):
        loss = []
        for model in self.models:
            loss.append(
                model.get_loss(states, actions, targets).item()
            )
        return np.mean(loss)

    def sample(self, data, batch_size=256):
        return self.models[0].sample(data, batch_size)
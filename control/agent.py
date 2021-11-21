import torch
class Agent:
    def __init__(self, env, model):
        self.model = model
        self.action_space = env.action_space

    # A placeholder function for any resetting of any model if necessary
    def reset(self):
        pass

    # The function that chooses the best action to take to most efficiently collect data
    def act(self, obs):
        return self.action_space.sample()

    # A function that trains all models within the agent
    def train(self, data, batch_size=64):
        states, actions, targets = self.model.sample(data, batch_size)
        loss = self.model.train_step(states, actions, targets)
        return loss

    # Sets normalization parameters of the model based on input data
    def normalize(self, states, actions):
        in_data = torch.cat([states, actions], -1)
        batch_mean, batch_std = torch.mean(in_data, 0), torch.std(in_data, 0)
        self.model.set_norm(batch_mean, batch_std)
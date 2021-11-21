from control.mpc import MPC
import numpy as np
import torch

class HillClimber(MPC):
    def __init__(self, model, env, rew_fn):
        super(HillClimber, self).__init__()
        self.rew_fn = rew_fn
        self.model = model
        self.env = env
        self.max_action = env.action_space.high
        self.act_dim = sum(env.action_space.shape)

    def evaluate(self, obs, act):
        x = torch.from_numpy(obs).to(torch.float32).cpu()
        a = torch.from_numpy(act).to(torch.float32).cpu()
        new_obs = self.model(x, a)
        r = self.rew_fn(new_obs)
        return r

    def act(self, obs):
        return self.hill_climb(obs)

    def hill_climb(self, obs, rand=False):
        epsilon = 0.001
        if rand:
            current_point = np.random.uniform(-1, 1, self.act_dim)
        else:
            current_point = np.zeros(self.act_dim)  # the 0 magnitude vector is common
        step_size = 100 * epsilon * np.ones(self.act_dim)  # a vector of all 1's is common
        acc = 1.2  # a value of 1.2 is common
        candidate = np.array([-acc, -1.0 / acc, 0.0, 1.0 / acc, acc])
        while True:
            before = self.evaluate(obs, current_point)
            for i in range(self.act_dim):
                best = -1
                best_score = -10000
                for j in range(5):
                    last_pt = current_point[i]
                    current_point[i] = current_point[i] + step_size[i] * candidate[j]
                    current_point[i] = max(current_point[i], -0.9)
                    current_point[i] = min(current_point[i], 0.9)

                    temp = self.evaluate(obs, current_point)
                    current_point[i] = last_pt
                    if temp > best_score:
                        best_score = temp
                        best = j
                if candidate[best] == 0:
                    step_size[i] = step_size[i] / acc
                else:
                    current_point[i] = current_point[i] + step_size[i] * candidate[best]
                    step_size[i] = step_size[i] * candidate[best]  # accelerate
            if (self.evaluate(obs, current_point) - before) < epsilon:
                return current_point

if __name__ == '__main__':
    import torch
    import pybullet
    import pybullet_envs
    import gym
    from body.walker import RealerWalkerWrapper
    from control.mpc import test_mpc
    model = torch.load('../trained/AntBulletEnv_model_0.pt').cpu()
    env = RealerWalkerWrapper(gym.make('AntBulletEnv-v0'))
    mpc = HillClimber(model, env, lambda s: s[0])
    test, baseline = test_mpc(env, mpc)
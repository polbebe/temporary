import numpy as np
from control.MPC.planners.parallelCEM import CEM
from control.MPC.sims.sim import Sim
import torch
class modelSim(Sim):
    def __init__(self, model):
        self.model = model
        self.state = None

    def init(self, obs):
        self.state = obs

    def sim_step(self, action):
        new_obs, logvar = self.model(torch.cat([self.state, action], -1))
        new_obs = new_obs.detach()
        if not torch.is_tensor(action):
            new_obs = new_obs.cpu().numpy()
        self.state = new_obs
        return new_obs

    def save(self):
        return self.state

    def load(self, state):
        self.state = state

def test_mpc(env, model, test_episodes=100):
    # state 0 of environment is assumed to be velocity in the forward x direction and is taken as reward
    sim = modelSim(model)
    mpc = CEM(sim, env.action_space, nsteps=10)
    mpc.p = 1 # Since we don't have an ensemble this removes the trajactory sampling

    # Baseline Reward
    # i = 0
    # env.reset()
    # baseline = []
    # ep_r = 0.0
    # while i < test_episodes:
    #     action = env.action_space.sample()
    #     new_obs, r, done, info = env.step(action)
    #     ep_r += r
    #     if done:
    #         baseline.append(ep_r)
    #         ep_r = 0.0
    #         env.reset()
    #         i += 1
    #         print('Base Episode '+str(i)+' done')

    # Test MPC Reward
    i = 0
    obs = env.reset()
    test = []
    ep_r = 0.0
    # action_mu = torch.zeros((mpc.nsteps, mpc.act_dim), device=device)
    # action_sigma = torch.ones((mpc.nsteps, mpc.act_dim), device=device)
    while i < test_episodes:
        next_mu, next_sigma = mpc.plan_move(obs)
        action = next_mu[0].cpu().numpy()

        # action_mu[:-1] = next_mu[1:]
        # action_sigma[:-1] = next_sigma[1:]
        new_obs, r, done, info = env.step(action)
        ep_r += r
        obs = new_obs
        if done:
            test.append(ep_r)
            ep_r = 0.0
            obs = env.reset()
            i += 1
            # print('Test Episode '+str(i)+' done')

    # print('Baseline Mean Ep R: '+str(np.mean(baseline)))
    # print('Baseline Std Ep R: '+str(np.std(baseline)))
    # print('Test MPC Mean Ep R: '+str(np.mean(test)))
    # print('Test MPC Std Ep R: '+str(np.std(test)))
    return test

if __name__ == '__main__':
    from train_models import MultiLinearModel
    from body.ant import RandAntEnv
    import numpy as np
    from body.walker import RealerWalkerWrapper
    env = RealerWalkerWrapper(RandAntEnv(render=False, percent_variation=0.05))

    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    max_action = 1
    env_string = 'AntBulletEnv'
    idx = 0

    runs = []
    for i in range(100):
        model = MultiLinearModel(state_dim, act_dim, state_dim, hid_size=256).to('cuda')
        # model = torch.load('../trained/Ant_model_'+str(i)+'.pt')
        test = test_mpc(env, model, test_episodes=1)
        runs.extend(test)

    print('Test MPC Mean Ep R: '+str(np.mean(runs)))
    print('Test MPC Std Ep R: '+str(np.std(runs)))
    print(runs)

from control.MPC.planners.parallelCEM import CEM
from control.MPC.sims.sim import modelSim
from control.agent import Agent
from body.PinkPanther.PinkPantherEnv import EasyPinkPantherEnv, HardPinkPantherEnv, RandPinkPantherEnv
from networking import NetEnv
# from models.test_ensemble import Ensemble
from models.seq_ensemble import Ensemble
import torch
import numpy as np
import time
import sys
import pickle as pkl
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_ep_len = 100


def val_set(Env, episodes, max_action=1):
    env = Env()
    i = 0
    states, actions, next_states = [], [], []
    episode_step = 0
    obs = env.reset()
    while i < episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        # data.append([obs, max_action * action, r, new_obs, done, episode_step])
        states.append(obs)
        actions.append(action*max_action)
        next_states.append(new_obs)
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            i += 1
    return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(next_states, dtype=np.float32))

def train_model(steps, env, agent, data=[], batch_size=64):
    obs = env.reset()

    for i in range(steps):
        action = agent.act(obs)
        new_obs, r, done, info = env.step(action)
        data.append((obs, action, new_obs))
        obs = new_obs
        if len(data) > batch_size:
            loss = agent.train(data, batch_size)
    return agent, loss, data

def plan_episode(env, planner):
    ep_r, ep_l = 0, 0
    done = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_mu = torch.zeros((planner.nsteps, planner.act_dim), device=device)
    action_sigma = torch.ones((planner.nsteps, planner.act_dim), device=device)

    obs = env.reset()
    obs_hist = [obs]
    start = time.time()

    pred_ep_r = 0
    pred_rs = []
    real_rs = []
    while not done:
        next_mu, next_sigma, pred_r = planner.plan_move(
            obs, action_mu, action_sigma, nsteps=planner.nsteps
        )
        pred_ep_r += pred_r
        action = next_mu[0].cpu().numpy()

        action_mu[:-1] = next_mu[1:]
        action_sigma[:-1] = next_sigma[1:]

        new_obs, r, done, info = env.step(action)
        obs = new_obs
        sys.stdout.write("Step: {} in {:.2f} seconds\t\tRew So Far: {:.2f} \r".format(ep_l, time.time() - start, ep_r))
        ep_r += r
        ep_l += 1

        done = ep_l >= max_ep_len

        pred_rs.append(pred_r)
        real_rs.append(r)
        obs_hist.append(obs)
    return obs_hist, real_rs, pred_rs

def train_test(train_steps, Env, Net, trials, runs, horizon, path, data_path, rew_fn):
    valid = val_set(Env, 5)
    results, losses = dict(), dict()
    envs = []
    for i in range(trials):
        env = Env()
        env.load_config(data_path+env_str+'_train_config_'+str(i))
        envs.append(env)
    pop_size = 5
    n_elites = 5

    # We assume all envs have the same state and action dimensions
    state_dim, act_dim = envs[0].observation_space.shape[0], envs[0].action_space.shape[0]

    # models = [Ensemble([Net(state_dim, act_dim, state_dim)]) for _ in range(trials)]
    models = [Ensemble([Net(state_dim, act_dim) for _ in range(pop_size)], pop_size, n_elites) for i in range(trials)]
    # models = [Ensemble(state_dim, act_dim, pop_size=pop_size, n_elites=n_elites, Network=BNN) for _ in range(trials)]
    agents = [Agent(envs[i], models[i].to(device)) for i in range(trials)]
    datas = [[] for _ in range(trials)]

    for i in range(len(train_steps)):
        start = time.time()
        results[train_steps[i]] = []
        losses[train_steps[i]] = []
        for j in range(trials):
            pred_rs, real_rs = [], []
            if train_steps[i] > 0:
                agents[j], train_loss, datas[j] = train_model(100, envs[j], agents[j], datas[j])
                val_loss = models[j].validation_loss(*valid)
                losses[train_steps[i]].append(val_loss)
                # losses[train_steps[i]].append(loss)
                pkl.dump(losses, open(path+'base_losses.pkl', 'wb+'))
            planner = CEM(modelSim(models[j]), envs[j].action_space, nsteps=horizon, rew_fn=rew_fn)
            print(str(train_steps[i]) +' Model trained in '+str(round(time.time()-start,3))+'s                                              ')
            for _ in range(runs):
                obs_hist, ep_r, pred_ep_r = plan_episode(envs[j], planner)
                real_rs.extend(ep_r)
                pred_rs.extend(pred_ep_r)
            results[train_steps[i]].append((real_rs, pred_rs))
            pkl.dump(results, open(path+'base_MPC_results.pkl', 'wb+'))
        assert train_steps[i] == np.mean([len(d) for d in datas])
        print(str(train_steps[i]) +' Finished in '+str(round(time.time()-start,3))+'s')
    return results, losses

def transfer_test(model_ids, model_path, data_path, Env, env_str, runs, horizon, save_path, rew_fn):
    results, losses = dict(), dict()
    start = time.time()
    valid = val_set(Env, 5)
    for id in model_ids:
        real_rs, pred_rs = [], []
        model = torch.load(model_path+env_str+'_model_'+str(id)+'.pt').to(device)
        # ensemble = Ensemble([model], 1, 1)
        env = Env()
        env.load_config(data_path+env_str+'_train_config_'+str(id))
        planner = CEM(modelSim(model), env.action_space, nsteps=horizon, rew_fn=rew_fn)
        scores = []
        for _ in range(runs):
            obs_hist, ep_r, pred_ep_r = plan_episode(env, planner)
            real_rs.extend(ep_r)
            pred_rs.extend(pred_ep_r)
            scores.append(sum(ep_r))
        results[id*100] = [(real_rs, pred_rs)]
        losses[id*100] = [model.validation_loss(*valid)]
        pkl.dump(losses, open(save_path+'multi_losses.pkl', 'wb+'))
        pkl.dump(results, open(save_path+'multi_MPC_results.pkl', 'wb+'))
        print(str(id)+' in '+str(round(time.time()-start,3))+'s with score '+str(np.mean(scores))+' +/- '+str(np.std(scores))+'                                              ')
    return results

if __name__ == '__main__':
    from models.bnn import BNN
    from models.linear import Linear
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--steps', type=int, default=0)  # total training steps
    parser.add_argument('--ep-len', type=int, default=100)  # steps per episode
    parser.add_argument('--trials', type=int, default=5)  # Number of different models to test over
    parser.add_argument('--runs', type=int, default=20)  # Number of times to run MPC per trial
    parser.add_argument('--horizon', type=int, default=1)  # MPC lookahead horizon
    parser.add_argument('--gpu', type=int, default=0)  # MPC lookahead horizon
    parser.add_argument('--model-path', type=str, default='trained')
    parser.add_argument('--data-path', type=str, default='data/PinkPanther')
    parser.add_argument('--env', type=str, default='rand')
    parser.add_argument('--path', type=str, default='./')  # Path to save logs to
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    if args.model_path[-1] != '/':
        args.model_path += '/'
    if args.data_path[-1] != '/':
        args.data_path += '/'

    env_str = args.env.lower()
    if env_str == 'base':
        from body.PinkPanther.PinkPantherEnv import PinkPantherEnv
        Env = PinkPantherEnv
    elif env_str == 'rand':
        Env = RandPinkPantherEnv
    elif env_str == 'hard':
        Env = HardPinkPantherEnv
    elif env_str == 'easy':
        Env = EasyPinkPantherEnv
    else:
        from body.PinkPanther.pinkpanther import PinkPantherEnv
        env_str = 'base'
        Env = PinkPantherEnv

    print('Training on '+env_str)
    args.data_path += env_str + '/'
    args.model_path += env_str + '/'

    rew_fn = lambda x: x[0] - 0.5 * x[1]
    # Baseline Test
    if args.steps > 0:
        train_steps = np.arange((args.steps/args.ep_len)+1)*args.ep_len
        train_test(train_steps, Env, BNN, args.trials, args.runs, args.horizon, args.path, args.data_path, rew_fn)
    # Testing Transferred Node Models
    elif args.model_path is not None:
        ids = np.arange(100)
        transfer_test(ids, args.model_path, args.data_path, Env, env_str, args.runs, args.horizon, args.path, rew_fn)

    print('')
    print('Finished')
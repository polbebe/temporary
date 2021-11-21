from body.walker import RealerWalkerWrapper
from body.ant import RandAntEnv
from body.dog import RandDogEnv
from body.spindra import RandSpindraEnv
from body.crawler import RandCrawlerEnv
from body.worm import RandWormEnv
import numpy as np

from multiprocessing import Process, Queue

# Multiprocessing function to be called
def run_process(q_in, q_out):
    id = q_in.get()
    print('Process '+str(id)+' started.')
    while True:
        obj = q_in.get()
        if obj is None:
            break
        Env, train_episodes, max_action, env_string, idx = obj
        save_env(Env, train_episodes, max_action, env_string, idx)
        q_out.put(id)
    print('Process '+str(id)+' ended.')

# Creating a train and validation set to be saved
def save_env(Env, train_episodes, max_action, env_string, idx):
    env = RealerWalkerWrapper(Env(render=False, percent_variation=0.05))
    # env = Env(render=False, percent_variation=0.05)
    print(idx)
    i = 0
    train = []
    episode_step = 0
    obs = env.reset()
    while i < train_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        # action = np.zeros(env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        train.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            episode_reward = 0.0
            i += 1
    valid = []
    i = 0
    episode_step = 0
    obs = env.reset()
    while i < train_episodes/4:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            episode_reward = 0.0
            i += 1
    np.save('data/'+env_string+'_train_' + str(len(train)) + '_'+str(idx)+'.npy', train)
    np.save('data/'+env_string+'_valid_' + str(len(valid)) + '_'+str(idx)+'.npy', valid)
    print('data/'+env_string+'_train_' + str(len(train)) + '_'+str(idx)+'.npy dumped')
    print('data/'+env_string+'_valid_' + str(len(valid)) + '_'+str(idx)+'.npy dumped')
    print('')

# Starting the parallel processes
def gen_data(Env, env_string):
    print('Ready')
    train_episodes = 100
    max_action = 1
    train_envs = 100
    n_cpu = 8
    import os
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Creating the queues
    Qs = []
    Ps = []
    busy = []
    for i in range(n_cpu):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=run_process, args=(q_in,q_out))
        p.start()
        q_in.put(i)
        Qs.append((q_in,q_out))
        Ps.append(p)
        busy.append(False)
    idx = 0
    # Adding all of the environment data to the queues
    while idx < train_envs:
        for i in range(len(busy)):
            if busy[i] == False:
                Qs[i][0].put((Env, train_episodes, max_action, env_string, idx))
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

    print(env_string+' Done')

# Separate testing code, unused
def test_env(Env, render=True):
    env = RealerWalkerWrapper(Env(render=render, percent_variation=0.05))
    # print(env.action_space.shape)
    # print(env.observation_space.shape)
    train_episodes = 100
    max_action = 1
    i = 0
    train = []
    episode_step = 0
    obs = env.reset()
    while i < train_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        # action = np.zeros(env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        train.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            i += 1

if __name__ == '__main__':
    # save_env(RandAntEnv, 100, 1, 'test', 0)
    gen_data(RandAntEnv, 'Ant')
    gen_data(RandDogEnv, 'Dog')
    gen_data(RandSpindraEnv, 'Spindra')
    gen_data(RandCrawlerEnv, 'Crawler')
    # gen_data(RandWormEnv, 'Worm')

    # test_env(RandAntEnv)
    # test_env(RandDogEnv)
    # test_env(RandSpindraEnv)
    # test_env(RandCrawlerEnv)
    # test_env(RandWormEnv)

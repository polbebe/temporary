import numpy as np
from multiprocessing import Process, Queue

# Multiprocessing function to be called
def run_process(q_in, q_out, path):
    id = q_in.get()
    print('Process '+str(id)+' started.')
    while True:
        obj = q_in.get()
        if obj is None:
            break
        Env, train_episodes, max_action, env_string, idx = obj
        save_env(Env, train_episodes, max_action, env_string, idx, path)
        q_out.put(id)
    print('Process '+str(id)+' ended.')

# Creating a train and validation set to be saved
def save_env(Env, train_episodes, max_action, env_string, idx, path):
    np.random.seed(idx)
    env = Env()
    print(idx)
    i = 0
    train = []
    episode_step = 0
    obs = env.reset()
    while i < train_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        train.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            i += 1
    env.save_config(path+env_string+'_train_config_'+str(idx))
    np.save(path+env_string+'_train_' + str(len(train)) + '_'+str(idx)+'.npy', train)
    print(path+env_string+'_train_' + str(len(train)) + '_'+str(idx)+' dumped')
    print('')
    env.close()

# Starting the parallel processes
def gen_data(Env, env_string, path, train_envs=100, train_episodes=10, n_cpu=8):
    print('Ready')
    max_action = 1
    import os
    if not os.path.isdir(path):
        os.mkdir(path)

    # Creating the queues
    Qs = []
    Ps = []
    busy = []
    for i in range(n_cpu):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=run_process, args=(q_in,q_out, path))
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

if __name__ == '__main__':
    import argparse
    from body.PinkPanther.PinkPantherEnv import HardPinkPantherEnv, EasyPinkPantherEnv, RandPinkPantherEnv, PinkPantherEnv
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpus', type=int, default=8)  # number of CPUs
    parser.add_argument('--envs', type=int, default=100000)  # number of envs to test
    parser.add_argument('--env', type=str, default='rand')
    args = parser.parse_args()
    # Training Data
    print('Generation Train Data')
    path = 'data/PinkPanther/'+args.env+'/'
    if args.env == 'easy':
        print('Generating '+args.env)
        gen_data(EasyPinkPantherEnv, args.env, path, train_envs=args.envs, n_cpu=args.cpus)
    elif args.env == 'hard':
        print('Generating '+args.env)
        gen_data(HardPinkPantherEnv, args.env, path, train_envs=args.envs, n_cpu=args.cpus)
    elif args.env == 'rand':
        print('Generating '+args.env)
        gen_data(RandPinkPantherEnv, args.env, path, train_envs=args.envs, n_cpu=args.cpus)
    elif args.env == 'base':
        print('Generating '+args.env)
        gen_data(PinkPantherEnv, args.env, path, train_envs=args.envs, n_cpu=args.cpus)
    # print('Generating Rand')
    # gen_data(RandPinkPantherEnv, 'rand', path, train_envs=100000)


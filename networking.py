import gym
import socket
import numpy as np

class NetEnv(gym.Env):
    def __init__(self, PORT=8123):
        # Socket Conneciton
        # MAC find WiFi IP - ipconfig getifaddr en0
        HOST = '128.59.145.148'
        # Set up Socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        print('Waiting for connection[s]...')
        # Wait for client to join socket
        self.s.listen()
        self.conn, addr = self.s.accept()
        print('Connected by: ', addr)

        self.conn.sendall(b'Init')
        d = np.frombuffer(self.conn.recv(1024), dtype=np.int)
        obs_high = np.ones(d[0])
        act_high = np.ones(d[1])
        self.observation_space = gym.spaces.Box(high=obs_high, low=-obs_high)
        self.action_space = gym.spaces.Box(high=act_high, low=-act_high)

    def reset(self):
        state = np.frombuffer(self.conn.recv(1024), dtype=np.float32)
        obs, r, done = state[:-2], state[-2], state[-1] == 1
        return obs

    def step(self, action):
        a = action.astype(np.float32)
        self.conn.sendall(a.tobytes())
        state = np.frombuffer(self.conn.recv(1024), dtype=np.float32)
        obs, r, done = state[:-2], state[-2], state[-1] == 1
        return obs, r, done, {}

def get_action(state):
    return np.random.uniform(-1, 1, 8)

if __name__ == '__main__':
    import time
    env = NetEnv()
    obs = env.reset()
    steps = 0
    done = False
    T = []
    while not done:
        start = time.time()
        action = get_action(obs)
        obs, r, done, info = env.step(action)
        steps += 1
        T.append(time.time()-start)
    T = T[:-1]
    # print(T)
    print('Time: '+str(np.sum(T)))
    print('Time Per Step: '+str(round(1000*np.mean(T), 3))+' +/- '+str(round(1000*np.std(T), 3))+'ms')
from pybullet_envs.robot_bases import MJCFBasedRobot
import numpy as np

class WalkerBase(MJCFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim, power):
    MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.power = power
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    self._p.setGravity(0, 0, -9.8)
    for j in self.ordered_joints:
      j.reset_current_position(np.random.uniform(low=-0.1, high=0.1), 0)

    self.feet = [self.parts[f] for f in self.foot_list]
    self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
      j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            # z - self.initial_z,
            # np.sin(angle_to_target),
            # np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)
    # return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)
    return np.clip(np.concatenate([more] + [j]), -5, +5)

  def calc_potential(self):
    # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    # all rewards have rew/frame units and close to 1.0
    debugmode = 0
    if (debugmode):
      print("calc_potential: self.walk_target_dist")
      print(self.walk_target_dist)
      print("self.scene.dt")
      print(self.scene.dt)
      print("self.scene.frame_skip")
      print(self.scene.frame_skip)
      print("self.scene.timestep")
      print(self.scene.timestep)
    return -self.walk_target_dist / self.scene.dt

import gym
from gym import spaces
class RealerWalkerWrapper(gym.Env):
    def __init__(self, env_in):
        self.env = env_in
        # obs_scale = 5
        obs_scale = 1
        act_scale = 1
        self.action_space = self.env.action_space
        # self.action_space.high *= act_scale
        # self.action_space.low *= act_scale

        self.front = 3 # first 3 need to be skipped as they are angle to target, 2nd 3 (totaling 6) account for velocity
        self.back = 4 # cutting out the feet contacts
        obs_ones = obs_scale*np.ones(shape=(self.env.observation_space.shape[0]-self.front-self.back,))
        self.observation_space = spaces.Box(high=self.env.observation_space.high[self.front:-self.back],
                                            low=self.env.observation_space.low[self.front:-self.back])
        # self.observation_space = self.env.observation_space
        # State Summary (dim=25):
        # state[0] = vx
        # state[1] = vy
        # state[2] = vz
        # state[3] = roll
        # state[4] = pitch
        # state[5 to -4] = Joint relative positions
        #    even elements [0::2] position, scaled to -1..+1 between limits
        #    odd elements  [1::2] angular speed, scaled to show -1..+1
        # state[-4 to -1] = feet contacts
        self.timestep = 0
        self.max_time = 100

    def reset(self):
        obs = self.env.reset()
        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        obs =  np.clip(obs[self.front:-self.back], -5, +5)
        # obs = obs[self.front:-self.back]
        self.timestep = 0
        self.ep_rew = 0
        return obs

    def step(self, action):
        new_obs, _, done, info = self.env.step(action)
        # r = (new_obs[3]/0.3)/60 # x velocity
        # r = new_obs[4] # y velocity
        r = new_obs[3]

        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        new_obs =  np.clip(new_obs[self.front:-self.back], -5, +5)
        # new_obs = new_obs[self.front:-self.back]
        self.timestep += 1
        done = self.timestep >= self.max_time
        self.ep_rew += r
        info = {}
        if done:
            info['episode'] = {}
            info['episode']['r'] = self.ep_rew
            info['episode']['l'] = self.timestep
        return new_obs, r, done, info

    def reset_raw(self):
        obs = self.env.reset()
        return obs
    def step_raw(self, action):
        new_obs, r, done, info = self.env.step(action)
        return new_obs, r, done, info
    def render(self, mode='human'):
        return self.env.render(mode)
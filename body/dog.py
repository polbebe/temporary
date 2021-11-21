from math import sin, cos, pi
import numpy as np
import os
import datetime
import time
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from body.walker import WalkerBase

class DogXMLBuilder:
    def __init__(self, legs=4, femur_len=0.28284271247461906, tibia_len=0.5656854249492381, body_volume=0.25):
        self.body = '''<mujoco model="dog">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1.5 0.1 0.1" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <worldbody>
    <body name="torso" pos="0 0 1">
            <geom fromto="-.5 0 0 .5 0 0" name="torso_geom" size="[body_volume]" type="capsule"/>
    </body>
  </worldbody>
  <actuator>
  </actuator>
</mujoco>
'''
        self.leg_def = \
        '''
      <body name="leg_[n]" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 [femur_x] [femur_y] [femur_z]" name="aux_[n]_geom" size="0.08" type="capsule" rgba=".8 .5 .3 1"/>
        <body name="aux_[n]" pos="[femur_x] [femur_y] [femur_z]">
          <joint axis="1 -1 0" name="hip_[n]" pos="0.0 0.0 0.0" range="-60 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 [femur_x] 0 [femur_z]" name="hip_[n]_geom" size="0.08" type="capsule" rgba=".8 .5 .3 1"/>
          <body pos="[femur_x] 0 [femur_z]" name="foot_[n]">
            <joint axis="1 -1 0" name="knee_[n]" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 [tibia_x] 0.0 [tibia_z]" name="knee_[n]_geom" size="0.08" type="capsule" rgba=".8 .5 .3 1"/>
          </body>
        </body>
      </body>'''
        self.act_def = \
    '''
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_[n]" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_[n]" gear="150"/>'''
        self.n = legs
        self.femur_r = femur_len
        self.tibia_r = tibia_len
        self.body_volume = body_volume

    def get_leg_coords(self):
        div = 2*pi/self.n
        thetas = [div*(i+1)-pi/4 for i in range(self.n)]
        femur_coords = []
        tibia_coords = []
        for theta in thetas:
            femur_coords.append((round(self.femur_r*cos(theta), 5), round(self.femur_r*sin(theta), 5)))
            tibia_coords.append((round(self.tibia_r*cos(theta), 5), round(self.tibia_r*sin(theta), 5)))
        return femur_coords, tibia_coords

    def build(self, dst_path=None):
        # self.datetime_str = 'test'
        if dst_path is None:
            dst_path = './dog_'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))+'.xml'
        dest_f = open(dst_path, 'w+')
        temp_s = self.body.replace('[body_volume]', str(self.body_volume))
        temp_split = temp_s.split('<geom fromto="-.5 0 0 .5 0 0" name="torso_geom" size="'+str(self.body_volume)+'" type="capsule"/>')
        write_s = temp_split[0] + '<geom fromto="-.5 0 0 .5 0 0" name="torso_geom" size="'+str(self.body_volume)+'" type="capsule"/>'
        femor_coords, tibia_coords = self.get_leg_coords()

        femur_y = [0,0,0,0]
        for i in range(self.n):
            n_leg_def = self.leg_def.replace('[n]', str(i+1))
            n_leg_def = n_leg_def.replace('[femur_x]', str(femor_coords[i][0]))
            n_leg_def = n_leg_def.replace('[femur_y]', str(femor_coords[i][1]))
            n_leg_def = n_leg_def.replace('[femur_z]', str(-abs(femor_coords[i][1])))
            n_leg_def = n_leg_def.replace('[tibia_x]', str(tibia_coords[i][0]))
            n_leg_def = n_leg_def.replace('[tibia_y]', str(tibia_coords[i][1]))
            n_leg_def = n_leg_def.replace('[tibia_z]', str(-abs(tibia_coords[i][1])))
            write_s += n_leg_def
        act_split = temp_split[1].split('<actuator>')
        write_s += act_split[0]+'<actuator>'
        for i in range(self.n):
            write_s += self.act_def.replace('[n]', str(i+1))
        write_s += act_split[1]
        dest_f.write(write_s)
        dest_f.flush()
        dest_f.close()
        return write_s

class RandDog(WalkerBase):
    def __init__(self, legs=4, femur_len=0.28284271247461906, tibia_len=0.5656854249492381, body_volume=0.25):
        path = os.getcwd()
        file = path+'/dog_test.xml'
        builder = DogXMLBuilder(legs, femur_len, tibia_len, body_volume)
        self.xml = builder.build(file)
        self.foot_list = ['foot_'+str(i+1) for i in range(legs)]
        WalkerBase.__init__(self, file, "torso", action_dim=2*legs, obs_dim=5+4*legs, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1

class RandDogEnv(WalkerBaseBulletEnv):
    def __init__(self, render=False, legs=4, femur_len=0.28284271247461906, tibia_len=0.5656854249492381, body_volume=0.25,
    # def __init__(self, render=False, legs=4, femur_len=0.15, tibia_len=0.3, body_volume=0.25,
                 percent_variation=0.05):
        self.robot = RandDog(legs,
                             np.random.normal(femur_len, femur_len*percent_variation),
                             np.random.normal(tibia_len, tibia_len*percent_variation),
                             np.random.normal(body_volume, tibia_len*percent_variation))
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

    def sample_tasks(self, num_tasks):
        femur_len=0.28284271247461906
        tibia_len=0.5656854249492381
        body_volume=0.25
        legs = 4
        sd = 0.025
        tasks = [(legs, np.random.normal(femur_len, sd), np.random.normal(tibia_len, sd),
                             np.random.normal(body_volume, sd)) for i in range(num_tasks)]
        return tasks

    def reset_task(self, task):
        # self.robot = RandDog(*task)
        # self.reset()
        self = RandDogEnv(False, *task)

if __name__ == '__main__':
    import pybullet as p

    p.connect(p.GUI)
    # plane = p.loadURDF("plane.urdf")
    # p.setGravity(0, 0, -9.8)
    p.setGravity(0, 0, 0)
    p.setTimeStep(1. / 500)
    p.setRealTimeSimulation(1)
    maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 20)
    env = RandDogEnv()
    print(env.action_space.shape)
    print(env.observation_space.shape)

    # env.reset()
    quadruped = p.loadMJCF('dog_test.xml')[0]
    index = 0
    jointIds = []
    paramIds = []
    jointDirections = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    jointAngles = [0, 0, 0, 0, 0, 0, 0, 0]
    jointOffsets = [0, 0, 0, 0, 0, 0, 0, 0]

    # for j in range(p.getNumJoints(quadruped)):
    #     p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    #     info = p.getJointInfo(quadruped, j)
    #     # print(info)
    #     jointName = info[1]
    #     jointType = info[2]
    #     if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    #         jointIds.append(j)

    for j in range(p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped, j)
        js = p.getJointState(quadruped, j)
        # print(info)
        jointName = info[1]
        jointType = info[2]
        if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
            jointIds.append(j)
            paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4,
                                                    (js[0] - jointOffsets[index]) / jointDirections[index]))
            index = index + 1
    p.setRealTimeSimulation(1)

    while (1):

        for i in range(len(paramIds)):
            c = paramIds[i]
            targetPos = p.readUserDebugParameter(c)
            maxForce = p.readUserDebugParameter(maxForceId)
            p.setJointMotorControl2(quadruped,jointIds[i],p.POSITION_CONTROL,jointDirections[i]*targetPos+jointOffsets[i], force=maxForce)
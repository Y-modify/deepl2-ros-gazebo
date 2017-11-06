#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import gym
import gym.spaces
from gym_gazebo.envs import gazebo_env
import numpy as np

import sys
import signal

from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam

from rl.core import Processor
#from rl.util import WhiteningNormalizer
from rl.agents import DDPGAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.keras_future import concatenate

import rl.callbacks
#import tensorflow as tf
#from keras.backend import tensorflow_backend

#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=config)
#tensorflow_backend.set_session(session)

class Clipper(Processor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)

class YamaXSimEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, "/yamax/src/yamax_gazebo/launch/world.launch")
	self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
	self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.joints = ["neck", "shoulder_right_x", "shoulder_right_z", "shoulder_left_x", "shoulder_left_z", "elbow_right", "elbow_left", "backbone_1", "backbone_2", "hip_joint_right_z", "hip_joint_right_x", "hip_joint_left_z", "hip_joint_left_x", "knee_right", "knee_left", "ankle_1_right", "ankle_1_left", "ankle_2_right", "ankle_2_left"]
        self.fail_threshold = 0.5
        self.success_threshold = 5
        self.dt = 0.1
        self.publishers = map(lambda j: rospy.Publisher('/yamax/'+j+'_position_controller', Float64, queue_size=10), self.joints)

	high_joints = np.array([1] * len(self.joints))
        # 行動空間 (+, 0, -の関節数乗) 1をもっと細かくするもありかも
        self.action_space = gym.spaces.Box(low=-high_joints, high=high_joints)

        high = np.array([1.57] * len(self.joints) + [1] + [10]) # 観測空間(state)の次元 (関節数+加速度+位置) とそれらの最大値(1)
        self.observation_space = gym.spaces.Box(low=-high, high=high) # 最小値は、最大値のマイナスがけ

    def n2a(self, x, n):
        sum = []
        while x:
            sum = [int(x % n)] + sum
            x -= x % n
            x /= n
        return sum

    def a2n(self, x, n):
        sum = 0
        for i, v in enumerate(x[::-1]):
            sum += v * n ** i
        return sum

    def read_sensors(self):
        acc = None
        while acc is None:
            try:
                acc = rospy.wait_for_message('/yamax/imu', Imu, timeout = 5).linear_acceleration.x
            except:
                pass

        pos = None
        while pos is None:
            try:
                pos = rospy.wait_for_message('/yamax/odom', Odometry, timeout = 5).pose.pose.position.x
            except:
                pass

        return acc, pos

    def unpause_simulation(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

    def pause_simulation(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    # action
    def _step(self, action):
        self.unpause_simulation()

        # action = n2a(action_n, 3)
        # actionを受け取り、次のstateを決定
        for (i, act) in enumerate(action):
            diff = (act - 1) * self.dt # acc: -0.1 0 0.1
            state = self.current_state[i] + diff
            self.current_state[i] = np.clip(state, -1.57, 1.57)
            self.publishers[i].publish(self.current_state[i])

        current_accel, current_position = self.read_sensors()

        fail = abs(current_accel) > self.fail_threshold
        success = current_position > self.success_threshold

        self.pause_simulation()

        if fail:
            reward = -1.0
        elif success:
            reward = 1.0
        else:
            # 時間経過ごとに負の報酬
            # 現在の加速度が0に近い(安定)ほど、へらない
            reward = -0.01 * abs(current_accel)

        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return np.array(self.current_state + [current_accel, current_position]), reward, fail or success, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):

        rospy.wait_for_service('/gazebo/reset_simulation') # reset env
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        self.current_state = [0] * len(self.joints)
        for pub in self.publishers:
            pub.publish(0) # reset degs

        self.unpause_simulation()

        current_accel, current_position = self.read_sensors()

        self.pause_simulation()

        return np.array(self.current_state + [current_accel, current_position])

env = YamaXSimEnv()
np.random.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# DQNのネットワーク定義
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = concatenate([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  processor=Clipper())
agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
agent.save_weights('ddpg_yamax_weights.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

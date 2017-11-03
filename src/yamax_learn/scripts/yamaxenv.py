#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import gym
import gym.spaces
from gym_gazebo.envs import gazebo_env
import numpy as np

from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import rl.callbacks

class YamaXSimEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, "/yamax/src/yamax_gazebo/launch/world.launch")
        self.joints = ["neck", "shoulder_right_x", "shoulder_right_z", "shoulder_left_x", "shoulder_left_z", "elbow_right", "elbow_left", "backbone_1", "backbone_2", "hip_joint_right_z", "hip_joint_right_x", "hip_joint_left_z", "hip_joint_left_x", "knee_right", "knee_left", "ankle_1_right", "ankle_1_left", "ankle_2_right", "ankle_2_left"]
        self.fail_threshold = 0.5
        self.success_threshold = 5
        self.dt = 0.1
        self.publishers = map(lambda j: rospy.Publisher('/yamax/'+j+'_position_controller', Float64, queue_size=10), self.joints)

        # 行動空間 (+, 0, -の関節数乗) 1をもっと細かくするもありかも
        self.action_space = gym.spaces.Discrete(3 ** len(self.joints))

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
    def _step(self, action_n):
        self.unpause_simulation()

        action = n2a(action_n, 3)
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
nb_actions = env.action_space.n

# DQNのネットワーク定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# experience replay用のmemory
memory = SequentialMemory(limit=50000, window_length=1)
# 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
        target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=300)
dqn.save_weights('dqn_weights.h5f', overwrite=True)
print(history.history)

dqn.test(env, nb_episodes=10, visualize=False)

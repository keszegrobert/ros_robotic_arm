#!/usr/bin/python

import gym
import numpy as np
from gym.envs.registration import register

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import sqrt
from random import random
from qlearn import QLearn
from qloader import QLoader
from datetime import datetime


N_DISCRETE_ACTIONS = 8  # aka |{B(-),B(+),S(-),S(+),E(-),E(+),W(-),W(+)}| = 8


class Owi535Model:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial',
                        anonymous=True, disable_signals=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print "============ Planning frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print "============ End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print(
            "============ Available Planning Groups:",
            robot.get_group_names()
        )

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        '''print "============ Printing robot state"
        print robot.get_current_state()
        print ""'''

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self, joint_goal):
        move_group = self.move_group
        current_pose = self.move_group.get_current_pose().pose
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        rospy.sleep(0.1)

        # For testing:
        current_joints = move_group.get_current_joint_values()
        # return all_close(joint_goal, current_joints, 0.01)

    def go_home(self):
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = 0
        joint_goal[2] = 0
        joint_goal[3] = 0
        self.go_to_joint_state(joint_goal)

    def go_diff(self, diff):
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0] += diff[0]
        joint_goal[1] += diff[1]
        joint_goal[2] += diff[2]
        joint_goal[3] += diff[3]
        # print("joint_goal: ", joint_goal)
        self.go_to_joint_state(joint_goal)

    def aerial_distance(self, goal):
        current = self.move_group.get_current_pose().pose
        dx = goal.position.x - current.position.x
        dy = goal.position.y - current.position.y
        dz = goal.position.z - current.position.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def get_new_goal(self):
        current = self.move_group.get_current_pose().pose
        current.position.x += random() * 0.2 - 0.1
        current.position.y += random() * 0.2 - 0.1
        current.position.z += random() * 0.2 - 0.1
        return current

    def get_observation(self):
        current = self.move_group.get_current_pose().pose
        return current


class Owi535Env1(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(gym.Env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        self.model = Owi535Model()
        self.previous_distance = 1.0
        self.pose_goal = geometry_msgs.msg.Pose()

    def get_reward(self, distance):
        if self.previous_distance > distance:
            if distance < 0.001:
                return 100000.0
            else:
                return 1.0 / distance
        else:
            return -1.0 / distance

    def step(self, action):
        # Execute one time step within the environment
        observation0 = self.model.get_observation()
        distance0 = self.model.aerial_distance(self.pose_goal)
        multiplier = 0.0001
        if distance0 < 0.01:
            multiplier = 0.5
        elif distance0 < 0.1:
            multiplier = 1.0
        else:
            multiplier = 5.0

        self.model.go_diff([
            multiplier * action[0] * 3.141 / 180.0,
            multiplier * action[1] * 3.141 / 180.0,
            multiplier * action[2] * 3.141 / 180.0,
            multiplier * action[3] * 3.141 / 180.0,
        ])

        observation = self.model.get_observation()
        distance = self.model.aerial_distance(self.pose_goal)
        reward = self.get_reward(distance)
        done = distance < 0.002 or distance > 0.5
        info = "{}".format(round(distance, 4))
        self.previous_distance = distance
        return observation, reward, done, info

    def reset(self):
        print("============ Resetting ===========")
        # Reset the state of the environment to an initial state
        self.pose_goal = self.model.get_new_goal()
        self.model.go_home()
        return self.model.get_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


def main():
    register(
        id='Owi535-v1',
        entry_point='training_simple:Owi535Env1',
    )
    env = gym.make('Owi535-v1')

    learner = QLearn(
        actions=range(env.action_space.n),
        alpha=0.1, gamma=0.7, epsilon=0.9
    )

    qloader = QLoader(learner)
    '''qloader.load("/home/thewhale/ws_moveit/20200120_201155.log")
    qloader.load("/home/thewhale/ws_moveit/20200120_212441.log")
    qloader.load("/home/thewhale/ws_moveit/20200120_212749.log")
    qloader.load("/home/thewhale/ws_moveit/20200120_213612.log")
    qloader.load("/home/thewhale/ws_moveit/20200121_201502.log")
    qloader.load("/home/thewhale/ws_moveit/20200122_220621.log")
    qloader.load("/home/thewhale/ws_moveit/20200124_203616.log")
    qloader.print_statistics()
    qloader.load("/home/thewhale/ws_moveit/20200126_221912.log")
    qloader.load("/home/thewhale/ws_moveit/20200127_193132.log")
    qloader.load("/home/thewhale/ws_moveit/20200128_191511.log")
    qloader.load("/home/thewhale/ws_moveit/20200129_194007.log")
    qloader.load("/home/thewhale/ws_moveit/20200131_002540.log")
    qloader.load("/home/thewhale/ws_moveit/20200131_202953.log")
    qloader.load("/home/thewhale/ws_moveit/20200201_185630.log")
    qloader.print_statistics()
    qloader.load("/home/thewhale/ws_moveit/20200209_210550.log")
    qloader.load("/home/thewhale/ws_moveit/20200210_195434.log")
    qloader.load("/home/thewhale/ws_moveit/20200211_194058.log")
    qloader.load("/home/thewhale/ws_moveit/20200212_194751.log")
    qloader.load("/home/thewhale/ws_moveit/20200213_210018.log")
    qloader.print_statistics()'''
    epsilon_discount = 0.999
    nepisodes = 500
    nsteps = 100
    highest_reward = 0
    now = datetime.now()
    logname = now.strftime("%Y%m%d_%H%M%S.stat")

    for i_episode in range(nepisodes):
        cumulated_reward = 0
        done = False
        if learner.epsilon > 0.05:
            learner.epsilon *= epsilon_discount

        obs = env.reset()
        state = ':'.join(map(str, [
            int((obs.position.x - env.pose_goal.position.x) * 1000),
            int((obs.position.y - env.pose_goal.position.y) * 1000),
            int((obs.position.z - env.pose_goal.position.z) * 1000)
        ]))
        for t in range(nsteps):
            env.render()
            # print(observation)
            action = learner.chooseAction(state)
            motor = action // 4
            direction = cmp(0, (action % 2) - 0.5)
            act = [0, 0, 0, 0]
            # print("motor: {}, action:{}, direction:{}".format(
            #     motor, action, direction))
            act[motor] = direction

            obs, reward, done, info = env.step(act)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ':'.join(map(str, [
                int((obs.position.x - env.pose_goal.position.x) * 1000),
                int((obs.position.y - env.pose_goal.position.y) * 1000),
                int((obs.position.z - env.pose_goal.position.z) * 1000)
            ]))
            learner.learn(state, action, reward, nextState)

            print("{}. reward: {} distance:{} //{}".format(
                t, reward, info, nextState))
            if done:
                break
            else:
                state = nextState
        print("Episode {} finished after {} timesteps, reward: {}".format(
            i_episode, t + 1, cumulated_reward
        ))
        with open(logname, "a") as f:
            f.write('{{"ep":"{}","steps":"{}","reward":"{}"}},\n'.format(
                i_episode, t + 1, cumulated_reward
            ))

    env.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

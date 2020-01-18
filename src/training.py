#!/usr/bin/env python

import gym
import numpy as np
from gym.envs.registration import register

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from math import sqrt
from random import random

N_DISCRETE_ACTIONS = 81  # aka |{-1,0,1}|^4 = 3^4 = 81


class Owi535Model:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
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

        # For testing:
        current_joints = move_group.get_current_joint_values()
        #return all_close(joint_goal, current_joints, 0.01)

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
        current.position.x += random() * 0.01 - 0.005
        current.position.y += random() * 0.01 - 0.005
        current.position.z += random() * 0.01 - 0.005
        return current

    def get_observation(self):
        current = self.move_group.get_current_pose().pose
        return [
            current.position.x,
            current.position.y,
            current.position.z
        ]


class Owi535Env(gym.Env):
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
        result = 0
        if self.previous_distance < distance:
            result += 1
        if distance < 0.01:
            result += 10
        if distance < 0.001:
            result += 100
        return result

    def step(self, action):
        # Execute one time step within the environment
        # print("+++++++++++++ Step {}++++++++++++++".format(action))
        motor_base = action % 3
        motor_shoulder = (action / 3) % 3
        motor_elbow = (action / 9) % 3
        motor_wrist = (action / 27) % 3
        # print("base: {}, shoulder: {}, elbow: {}, wrist:{}".format(
        #    motor_base - 1, motor_shoulder - 1,
        #    motor_elbow - 1, motor_wrist - 1
        # ))
        self.model.go_diff([
            (motor_base - 1) * 3.141 / 180.0,
            (motor_shoulder - 1) * 3.141 / 180.0,
            (motor_elbow - 1) * 3.141 / 180.0,
            (motor_wrist - 1) * 3.141 / 180.0,
        ])

        observation = self.model.get_observation()
        distance = self.model.aerial_distance(self.pose_goal)
        reward = self.get_reward(distance)
        done = distance < 0.001
        self.previous_distance = distance
        info = "{}".format(action)
        return observation, reward, done, info

    def reset(self):
        print("============ Resetting ===========")
        # Reset the state of the environment to an initial state
        self.pose_goal = self.model.get_new_goal()
        self.model.go_home()


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


def main():
    register(
        id='Owi535-v0',
        entry_point='training:Owi535Env',
    )
    env = gym.make('Owi535-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print("{}. reward: {}".format(t, reward))
            if done:
                print("Episode {} finished after {} timesteps".format(
                    i_episode, t + 1
                ))
                break
    env.close()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

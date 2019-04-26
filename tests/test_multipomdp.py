import unittest

import numpy as np
import numpy.random as rnd

import gym
import gym_pomdps


class MultiPOMDP_Test(unittest.TestCase):
    @unittest.skip
    def test_run(self):
        env = gym.make('POMDP-shopping_2.e-v0')
        nactions = env.action_space.n

        ntrajectories = 3
        env = gym_pomdps.MultiPOMDP(env, ntrajectories)

        nepisodes, nsteps = 10, 100
        for i in range(nepisodes):
            env.reset()
            for t in range(nsteps):
                actions = rnd.randint(nactions, size=ntrajectories)
                obss, rewards, dones, infos = env.step(actions)

    # MultiPOMDP does not support episodic POMDPs yet
    @unittest.expectedFailure
    def test_run_episodic(self):
        env = gym.make('POMDP-shopping_2.e-episodic-v0')
        nactions = env.action_space.n

        ntrajectories = 3
        env = gym_pomdps.MultiPOMDP(env, ntrajectories)

        nepisodes, nsteps = 1, 1000
        for i in range(nepisodes):
            env.reset()
            for t in range(nsteps):
                actions = rnd.randint(nactions, size=ntrajectories)
                obss, rewards, dones, infos = env.step(actions)
                # print(actions, rewards, dones)

import unittest

import numpy as np
import numpy.random as rnd

import gym
import gym_pomdps


# TODO switch to testing the _functionsl methods


class Gym_MultiPOMDP_Test(unittest.TestCase):
    def test_run(self):
        ntrajectories = 5
        nepochs, nsteps = 10, 100

        env = gym.make('POMDP-shopping_2-v0')
        env = gym_pomdps.MultiPOMDP(env, ntrajectories)

        for _ in range(nepochs):
            env.reset()
            for _ in range(nsteps):
                actions = rnd.randint(env.action_space.n, size=ntrajectories)
                obs, rewards, dones, infos = env.step(actions)

                if dones.any():
                    raise Exception('Non-episodic Environment should not end '
                                    f'(dones={dones})')


    def test_run_episodic(self):
        ntrajectories = 5
        nsteps = 1000

        env = gym.make('POMDP-shopping_2-episodic-v0')
        env = gym_pomdps.MultiPOMDP(env, ntrajectories)
        env.reset()
        for _ in range(nsteps):
            actions = rnd.randint(env.action_space.n, size=ntrajectories)
            obs, rewards, dones, infos = env.step(actions)

            if dones.all():
                break
        else:
            raise Exception(f'Episodic Environment did not end (dones={dones})')

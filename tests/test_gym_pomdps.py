import unittest

import gym
import gym_pomdps


class GymPOMDPs_Test(unittest.TestCase):
    def test_list_pomdps(self):
        pomdps = gym_pomdps.list_pomdps()

        self.assertTrue(len(pomdps) > 0)
        for pomdp in pomdps:
            self.assertTrue(pomdp.startswith('POMDP'))

    @unittest.skip
    def test_make(self):
        envs = {}
        for pomdp in gym_pomdps.list_pomdps():
            try:
                env = gym.make(pomdp)
            except MemoryError:  # some POMDPs are too big
                pass
            else:
                envs[pomdp] = env

    def test_seed(self):
        env = gym.make('POMDP-loadunload-v0')

        seed = 17
        actions = list(range(env.action_space.n)) * 10

        env.seed(seed)
        env.reset()
        outputs = list(map(env.step, actions))

        env.seed(seed)
        env.reset()
        outputs2 = list(map(env.step, actions))

        self.assertEqual(outputs, outputs2)

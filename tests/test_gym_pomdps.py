import random
import unittest

import gym
import gym_pomdps

# TODO switch to testing the _functionsl methods


class Gym_POMDP_Test(unittest.TestCase):
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

    def test_run(self):
        env = gym.make('POMDP-shopping_2-v0')
        env.reset()

        for i in range(100):
            a = random.randint(0, env.action_space.n - 1)
            o, r, done, info = env.step(a)

    def test_run_episodic(self):
        env = gym.make('POMDP-shopping_2-episodic-v0')
        env.reset()

        for i in range(100):
            a = random.randint(0, env.action_space.n - 1)
            o, r, done, info = env.step(a)
            if done:
                break

        if not done:
            raise Exception('Episodic Environment did not end')

    def test_seed(self):
        seed = 17

        env = gym.make('POMDP-loadunload-v0')
        actions = list(range(env.action_space.n)) * 10

        # run environment multiple times with same seed
        outputs = []
        for _ in range(2):
            env.seed(seed)
            env.reset()
            output = list(map(env.step, actions))
            outputs.append(output)

        self.assertEqual(*outputs)


if __name__ == '__main__':
    unittest.main()

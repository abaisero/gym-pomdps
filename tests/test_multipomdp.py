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
                    raise Exception(
                        'Non-episodic Environment should not end '
                        f'(dones={dones})'
                    )

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

    def test_seed(self):
        seed = 17
        ntrajectories, nsteps = 5, 10

        env = gym.make('POMDP-loadunload-v0')
        env = gym_pomdps.MultiPOMDP(env, ntrajectories)
        actions = rnd.randint(env.action_space.n, size=(nsteps, ntrajectories))

        # run environment multiple times with same seed
        outputs = []
        for _ in range(2):
            env.seed(seed)
            env.reset()
            output = list(map(env.step, actions))
            outputs.append(output)

        outputs1, outputs2 = outputs
        for (o1, r1, done1, info1), (o2, r2, done2, info2) in zip(*outputs):
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(done1, done2)
            np.testing.assert_equal(info1, info2)

    def test_consistency(self):
        seed = 17
        ntrajectories = 1  # single sample necessary to control randomness
        nsteps = 10

        env = gym.make('POMDP-loadunload-v0')
        actions = rnd.randint(env.action_space.n, size=nsteps)

        env.seed(seed)
        env.reset()
        outputs1 = list(map(env.step, actions))

        env = gym_pomdps.MultiPOMDP(env, ntrajectories)
        actions = actions.reshape(-1, 1)

        env.seed(seed)
        env.reset()
        outputs2 = list(map(env.step, actions))

        outputs = zip(outputs1, outputs2)
        for (o1, r1, done1, info1), (o2, r2, done2, info2) in outputs:
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(done1, done2)
            np.testing.assert_equal(info1, info2)


if __name__ == '__main__':
    unittest.main()

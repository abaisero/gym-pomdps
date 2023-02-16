import gym
import numpy as np
import numpy.random as rnd
import pytest

from gym_pomdps.wrappers import BatchPOMDP


@pytest.mark.parametrize(
    'env_id,batch_size',
    [
        ('POMDP-shopping_2-continuing-v0', 5),
        ('POMDP-shopping_2-episodic-v0', 5),
    ],
)
def test_functional(env_id: str, batch_size: int):
    env = gym.make(env_id)
    env = BatchPOMDP(env, batch_size)
    for _ in range(20):
        s, o = env.reset_functional()
        assert isinstance(s, np.ndarray)
        assert s.dtype == int
        assert ((s >= 0) & (s < env.state_space.n)).all()
        assert isinstance(o, np.ndarray)
        assert o.dtype == int
        assert (o == env.observation_space.n - 1).all()

        s = rnd.randint(env.state_space.n, size=batch_size)
        a = rnd.randint(env.action_space.n, size=batch_size)

        s1, o, r, done, info = env.step_functional(s, a)
        assert isinstance(s1, np.ndarray)
        assert s1.dtype == int
        assert isinstance(o, np.ndarray)
        assert o.dtype == int
        assert isinstance(r, np.ndarray)
        assert r.dtype == float
        assert isinstance(done, np.ndarray)
        assert done.dtype == bool

        assert (s1[done] == -1).all()
        assert ((s1[~done] >= 0) & (s1[~done] < env.state_space.n)).all()
        assert ((o >= 0) & (o < env.observation_space.n)).all()
        assert set(r).issubset(env.rewards_dict.keys())
        assert info is None or isinstance(info, dict)

        s = np.full([batch_size], -1)
        a = rnd.randint(env.action_space.n, size=batch_size)
        with pytest.raises(ValueError):
            env.step_functional(s, a)

        s = np.full([batch_size], -1)
        a = np.full([batch_size], -1)
        s1, o, r, done, info = env.step_functional(s, a)
        assert isinstance(s1, np.ndarray)
        assert s1.dtype == int
        assert isinstance(o, np.ndarray)
        assert o.dtype == int
        assert isinstance(r, np.ndarray)
        assert r.dtype == float
        assert isinstance(done, np.ndarray)
        assert done.dtype == bool
        assert info is None or isinstance(info, dict)
        assert (s1 == -1).all()
        assert (o == -1).all()
        assert (r == 0.0).all()
        assert done.all()


@pytest.mark.parametrize(
    'env_id,batch_size',
    [
        ('POMDP-shopping_2-continuing-v0', 5),
        ('POMDP-shopping_2-episodic-v0', 5),
    ],
)
def test_run(env_id: str, batch_size: int):
    env = gym.make(env_id)
    env = BatchPOMDP(env, batch_size)
    for _ in range(20):
        done = np.full(batch_size, False)
        env.reset()
        for _ in range(100):
            a = rnd.randint(env.action_space.n, size=batch_size)
            a[done] = -1
            o, r, done1, info = env.step(a)

            assert isinstance(o, np.ndarray)
            assert isinstance(r, np.ndarray)
            assert isinstance(done1, np.ndarray)
            assert o.dtype == int
            assert r.dtype == float
            assert done1.dtype == bool
            assert info is None or isinstance(info, dict)

            assert (o[done & done1] == -1).all()
            assert (r[done & done1] == 0.0).all()
            assert (r[done & done1] == 0.0).all()
            assert done1[done].all()
            assert (
                (o[~done & done1] >= 0) & (o[~done & done1] <= env.observation_space.n)
            ).all()
            assert set(r[~done & done1]).issubset(env.rewards_dict.keys())
            done = done1


@pytest.mark.parametrize('env_id', ['POMDP-tiger-continuing-v0'])
@pytest.mark.parametrize(
    'seed1,seed2',
    [
        (17, 17),
        (17, 18),
        (18, 18),
    ],
)
def test_reproducibility(env_id: str, seed1: int, seed2: int):
    batch_size, num_steps = 20, 100

    env = gym.make(env_id)
    env = BatchPOMDP(env, batch_size)
    actions = rnd.randint(env.action_space.n, size=(num_steps, batch_size))

    env.seed(seed1)
    outputs1 = [env.reset()] + list(map(env.step, actions))

    env.seed(seed2)
    outputs2 = [env.reset()] + list(map(env.step, actions))

    expected: bool = seed1 == seed2
    if expected:
        np.testing.assert_equal(outputs1, outputs2)
    else:
        with pytest.raises(AssertionError):
            np.testing.assert_equal(outputs1, outputs2)


# checks that the batched implementation matches the underlying implementation
@pytest.mark.parametrize('env_id', ['POMDP-loadunload-continuing-v0'])
@pytest.mark.parametrize('seed', [7, 17])
def test_consistency(env_id: str, seed: int):
    num_steps = 10

    env = gym.make(env_id)
    actions = rnd.randint(env.action_space.n, size=num_steps)

    env.seed(seed)
    outputs1 = [env.reset()] + list(map(env.step, actions))

    env = BatchPOMDP(env, batch_size=1)
    actions = actions.reshape(-1, 1)

    env.seed(seed)
    outputs2 = [env.reset()] + list(map(env.step, actions))

    np.testing.assert_equal(outputs1, outputs2)

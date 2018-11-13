from pkg_resources import resource_listdir, resource_filename, resource_isdir
from .envs import POMDP
from .wrappers import MultiPOMDP
from .envs.registration import register, env_list


def list_pomdps():
    return list(env_list)


def is_pomdp(name):
    return (name.upper().endswith('.POMDP') and
            not resource_isdir('gym_pomdps.pomdps', name))


fnames = filter(is_pomdp, resource_listdir('gym_pomdps.pomdps', ''))
for fname in fnames:
    fpath = resource_filename('gym_pomdps.pomdps', fname)
    name = '.'.join(fname.split('.')[:-1])

    register(
        id=f'POMDP-{name}-v0',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(path=fpath, episodic=False),
    )

    register(
        id=f'POMDP-{name}-episodic-v0',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(path=fpath, episodic=True),
    )

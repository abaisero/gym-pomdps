import re

from pkg_resources import resource_filename, resource_isdir, resource_listdir

from .envs import POMDP
from .envs.registration import env_list, register
from .wrappers import MultiPOMDP


def list_pomdps():
    return list(env_list)


def is_pomdp(name):
    return name.upper().endswith('.POMDP') and not resource_isdir(
        'gym_pomdps.pomdps', name
    )


fnames = filter(is_pomdp, resource_listdir('gym_pomdps.pomdps', ''))
for fname in fnames:
    fpath = resource_filename('gym_pomdps.pomdps', fname)
    name = '.'.join(fname.split('.')[:-1])  # remove .pomdp suffix
    version = 0

    # extract version if any
    m = re.fullmatch('(?P<name>.*)\.v(?P<version>\d+)', name)
    if m is not None:
        name, version = m['name'], m['version']

    register(
        id=f'POMDP-{name}-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(path=fpath, episodic=False),
    )

    register(
        id=f'POMDP-{name}-episodic-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(path=fpath, episodic=True),
    )

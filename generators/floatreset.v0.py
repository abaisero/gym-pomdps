#!/usr/bin/env python
import argparse
import copy

import one_to_one


def sfmt(s):
    return str(s.value)


def afmt(a):
    return a.value


def ofmt(o):
    return str(o.value)


def main():
    parser = argparse.ArgumentParser(description='FloatReset')
    parser.add_argument('--gamma', type=float, default=0.99)
    config = parser.parse_args()

    state_space = one_to_one.RangeSpace(5)

    actions = 'f', 'r'
    action_space = one_to_one.DomainSpace(actions)

    obs_space = one_to_one.RangeSpace(2)

    print(
        """# Float/Reset Environment;

# @inproceedings{littman_predictive_2002,
#     title = {Predictive representations of state},
#     booktitle = {Advances in neural information processing systems},
#     author = {Littman, Michael L. and Sutton, Richard S.},
#     year = {2002},
#     pages = {1555--1561},
# }

# State-space (5) : current position.

# Action-space (2) : `f` and `r`.

# Observation-space (2) : 0 and 1."""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {len(state_space)}')

    print(f'actions: {" ".join(afmt(a) for a in action_space.elems())}')

    print(f'observations: {len(obs_space)}')

    # START
    print()
    s = state_space.elem(value=0)
    print(f'start: {sfmt(s)}')

    # TRANSITIONS
    print()

    a = action_space.elem(value='r')
    s1 = state_space.elem(value=0)
    print(f'T: {afmt(a)}: *: {sfmt(s1)} 1.0')

    a = action_space.elem(value='f')
    for s in state_space.elems():
        s1 = copy.copy(s)
        try:
            s1.value -= 1
        except ValueError:
            pass
        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 0.5')

        s1 = copy.copy(s)
        try:
            s1.value += 1
        except ValueError:
            pass
        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 0.5')

    # OBSERVATIONS
    print()

    a = action_space.elem(value='f')
    print(f'O: * : * 1.0 0.0')

    a = action_space.elem(value='r')
    s1 = state_space.elem(value=0)
    print(f'OO: {afmt(a)} : {sfmt(s1)} : {sfmt(s1)} 0.0 1.0')

    # REWARDS
    print()
    a = action_space.elem(value='r')
    for s in state_space.elems():
        print(f'R: {afmt(a)}: {sfmt(s)}: *: * {s.value:.1f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import copy

import one_to_one


def pfmt(p):
    return str(p.value)


def sfmt(s):
    return f'f_{s.value.old}_t_{s.value.new}'


def afmt(a):
    return a.value


def ofmt(o):
    return str(o.value)


def is_adjacent(s):
    return abs(s.new - s.old) == 1 or s.new == s.old == 4 or s.new == 0


def main():
    parser = argparse.ArgumentParser(description='FloatReset')
    parser.add_argument('--gamma', type=float, default=0.99)
    config = parser.parse_args()

    pos_space = one_to_one.RangeSpace(5)

    state_space = one_to_one.NamedTupleSpace(old=pos_space, new=pos_space)
    state_space = one_to_one.SubSpace(state_space, is_adjacent)

    actions = 'float', 'reset_'
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

# Action-space (2) : `float` and `reset_`.

# Observation-space (2) : 0 and 1."""
    )

    print()
    print(f'# This specific file was generated with parameters:')
    print(f'# {config}')
    print()
    print(f'discount: {config.gamma}')
    print('values: reward')

    print(f'states: {" ".join(sfmt(s) for s in state_space.elems())}')

    print(f'actions: {" ".join(afmt(a) for a in action_space.elems())}')

    print(f'observations: {len(obs_space)}')

    # START
    print()
    s = state_space.elem(0)
    s.value = s.value._replace(old=0, new=0)
    print(f'start: {sfmt(s)}')

    # TRANSITIONS
    print()

    a = action_space.elem(value='reset_')
    for s in state_space.elems():
        s1 = state_space.elem(value=s.value._replace(old=s.value.new, new=0))
        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 1.0')

    a = action_space.elem(value='float')
    for s in state_space.elems():
        try:
            s1.value = s.value._replace(old=s.value.new, new=s.value.new - 1)
        except ValueError:
            s1 = state_space.elem(value=s.value._replace(old=s.value.new))

        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 0.5')

        try:
            s1.value = s.value._replace(old=s.value.new, new=s.value.new + 1)
        except ValueError:
            s1 = state_space.elem(value=s.value._replace(old=s.value.new))

        print(f'T: {afmt(a)}: {sfmt(s)}: {sfmt(s1)} 0.5')

    # OBSERVATIONS
    print()

    print(f'O: *: * 1.0 0.0')

    a = action_space.elem(value='reset_')
    s1 = state_space.elem(0)
    s1.value = s1.value._replace(old=0, new=0)
    print(f'O: {afmt(a)}: {sfmt(s1)} 0.0 1.0')

    # REWARDS
    print()
    a = action_space.elem(value='reset_')
    for s in state_space.elems():
        print(f'R: {afmt(a)}: {sfmt(s)}: *: * {s.value.new:.1f}')


if __name__ == '__main__':
    main()

# gym-pomdps

This repository contains gym environments for flat POMDPs which can be loaded
from the `.pomdp` file format.

## Installation

This package is dependent on the
[rl_parsers](https://github.com/abaisero/rl_parsers) package.  Install
`rl_parsers` before proceeding.

## Contents

The main contents of the repository are the `POMDP` environment, and the
`MultiPOMDP` wrapper.

### POMDP Environment

The POMDP environment receives a path to the `.pomdp` file, and boolean flag
indicating whether the POMDP should be considered episodic or continuing.

NOTE:  the episodic version is only supported if the POMDP file format makes
use of the custom `reset` keyword (see
[rl_parsers](https://github.com/abaisero/rl_parsers) for details);  if this is
not the case, the two versions are equivalent (and both continuing).

All the POMDPs in the `pomdps/` folder are registered under gym:
 * A continuing version under ID `POMDP-{name}-v0`; and
 * An episodic version under ID `POMDP-{name}-episodic-v0`.

### MultiPOMDP Wrapper

The MultiPOMDP Wrapper allows to run multiple indipendent instances of the same
POMDP at the same time, and is more efficient that running each instance one
after the other.  The wrapper receives a standard POMDP environment and the
number of independent instances to run.  The resulting step function receives
an array of actions and returns arrays of observations, rewards and dones.

NOTE:  This wrapper currently only supports non-episodic POMDPs.

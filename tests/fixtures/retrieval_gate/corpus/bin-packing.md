# Online 3D Bin Packing

In online 3D bin packing, cuboid boxes arrive one at a time and each must be placed in a container before the next box is seen. The goal is high volume utilisation without toppling the stack.

The container state is usually represented as a heightmap: a 2D grid storing the current stack height at each cell. A placement policy selects a position and orientation for the incoming box. Heuristics such as deepest-bottom-left work well, but deep reinforcement learning policies trained in simulation reach higher packing density.

Stability matters as much as density. A placement is rejected when the box would overhang its support by more than a threshold or when the centre of mass falls outside the support polygon. Fragile items add a constraint: heavy boxes must not be placed on top of them.

Benchmarks report utilisation (percentage of container volume filled) and the number of boxes placed before the episode ends.

# Project Log

## Week 1

Set up the robot, the gripper, and the camera. First grasp tests on boxes and bottles. The depth camera point cloud is noisy at the table edge. The planner moves the robot to a pre-grasp pose above the object before closing the gripper.

## Week 2

Packing tests with boxes in the container. The robot places boxes in a simple left-to-right order; some boxes overhang and fall. Tried a learned policy from simulation; it packs more boxes but the robot motion is jerky. Logged twenty runs.

## Week 3

Grocery tests: eggs, bottles, fruit, cartons. The gripper crushed one egg at default force. Lowered the force. The camera misses transparent bottles. The language model planner sometimes orders fragile items before heavy items, which is wrong for packing.

## Week 4

Bag tests with the bag stand. The robot drops items next to the bag when the bag opening is small. Tried pulling the bag rim with the gripper; the bag slips. Camera view from above helps to see the bag opening.

## Week 5

Simulation work: trained the packing policy with more varied box sizes and masses. The simulated robot succeeds far more often than the real robot. Started looking at why the real camera and the simulated camera differ.

## Week 6

Calibration problems: the gripper misses objects by about a centimetre in the same direction every time. The planner and the policy are fine in simulation. Planning to redo the camera setup and check the robot tool frame.

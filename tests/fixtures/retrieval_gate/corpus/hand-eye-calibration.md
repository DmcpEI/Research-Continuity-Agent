# Hand-Eye Calibration

Hand-eye calibration finds the rigid transform between a camera and the robot. In the eye-in-hand setup the camera is mounted on the gripper; in the eye-to-hand setup it is fixed in the workcell.

The robot moves to several poses while the camera observes a checkerboard or ArUco marker board. Each pair of poses gives an equation of the form AX = XB, where A is the gripper motion, B the camera motion, and X the unknown camera-to-gripper transform. Solvers such as Tsai-Lenz or Park-Martin estimate X from many pose pairs.

Poor calibration shows up as a constant offset between where the camera sees an object and where the gripper arrives. Rotation diversity between poses matters more than the number of poses.

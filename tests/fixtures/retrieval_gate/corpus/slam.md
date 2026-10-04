# Visual SLAM Notes

Simultaneous localisation and mapping (SLAM) estimates the camera trajectory while building a map of the environment. Feature-based visual SLAM tracks ORB keypoints between frames and triangulates map points.

Drift accumulates over time. Loop closure detects that the camera has returned to a previously mapped place, using a bag-of-words place recognition index, and adds a constraint between the two poses. Pose graph optimisation then distributes the correction along the trajectory.

Local bundle adjustment jointly refines recent keyframe poses and map point positions by minimising reprojection error. Monocular SLAM has an unknown scale; stereo or RGB-D cameras, or an IMU, recover metric scale.

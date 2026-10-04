# Grasp Planning from Point Clouds

Grasp planning chooses where a robot gripper should close on an object. Given a depth camera point cloud, the planner first segments the target object from the table plane with RANSAC, then samples candidate grasp poses on the object surface.

For a parallel-jaw gripper, an antipodal grasp places the two fingers on surfaces whose normals point in opposite directions. Candidates are scored by a learned grasp quality network that predicts the probability of a stable lift. The top candidate is checked for collisions with the scene before execution.

Suction grasps are an alternative for flat, rigid surfaces such as boxes and cartons. A suction cup needs a locally planar patch larger than the cup diameter; porous packaging or deformable bags often fail to seal. Many warehouse picking systems combine both: suction for cartons, parallel jaws for irregular items.

Common failure modes are transparent objects, which leave holes in the point cloud, and cluttered bins where the segmentation merges touching objects.

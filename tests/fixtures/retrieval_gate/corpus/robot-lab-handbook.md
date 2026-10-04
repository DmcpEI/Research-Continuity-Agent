# Robot Lab Handbook

This handbook collects everyday practice for the robot lab. It covers the robot arm, the gripper, the cameras, the packing station, and how to plan and log experiments. It is general guidance; the topic notes hold the details.

## The robot arm

The robot arm is a six-axis collaborative arm. Before moving the robot, check that the workcell is clear and that the emergency stop is within reach. Move the robot slowly when testing a new plan or a new policy. The robot arm controller runs on the lab workstation; the planner sends joint targets to it. Always home the robot at the end of the day.

## The gripper

The lab has a parallel gripper and a suction gripper. The gripper is swapped by hand; after swapping the gripper, update the tool frame in the robot controller. A gripper that closes too hard can crush fragile items such as eggs or soft fruit, so start with a low gripper force. When a grasp fails, log the object, the gripper, and the camera view.

## The cameras

Two cameras are mounted in the workcell: a depth camera above the table and a colour camera on the robot wrist. Keep the camera lenses clean. If the camera image looks shifted relative to the robot, report it; the camera may have moved. Camera drivers are restarted with the lab script. The depth camera struggles with shiny and transparent objects.

## The packing station

The packing station has a table, a container for boxes, and a stand that holds a bag open. Items for packing experiments come in a crate: boxes, bottles, cartons, eggs, fruit, and bags of rice. When packing groceries, the plan should put heavy items in first and fragile items last. Boxes must not overhang the container edge. Bags are packed with the bag stand; a bag that falls over ruins the run.

## Planning experiments

Every experiment needs a plan: the question, the robot setup, the policy or planner being tested, and the number of trials. A planner that uses a language model must have its prompt and model version logged. Planning in simulation first saves robot time; a policy trained in simulation must still be tested on the robot.

## Logging and data

Log every run: date, robot, gripper, camera, policy, planner, success or failure, and notes. Store camera images and depth data with the run. Use the shared spreadsheet for the run list and the experiment tracker for metrics. A run without a log did not happen.

## Safety and cleanup

Wear safety glasses near the robot. Do not leave items in the gripper. Put boxes, bags, and the crate back on the shelf. Switch off the cameras and the robot at the end of the day.

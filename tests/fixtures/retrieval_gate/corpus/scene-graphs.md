# 3D Scene Graphs for Task Planning

A 3D scene graph is a hierarchical representation of an environment: buildings contain floors, floors contain rooms, rooms contain objects, and objects carry attributes such as state and affordances. Edges encode containment and spatial relations.

For long-horizon task planning in large environments, a language model cannot read the full map. Instead the planner expands and collapses parts of the scene graph, searching only the rooms relevant to the instruction. This semantic search keeps the prompt small enough for the model context window.

Plans proposed by the language model are verified by a classical path planner and a scene graph simulator before execution; infeasible steps are fed back to the model for replanning. Dynamic scene graphs extend the idea by updating object nodes as the robot observes changes over time.

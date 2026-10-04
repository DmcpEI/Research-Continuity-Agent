# Sim-to-Real Transfer

Policies trained in simulation often fail on the real robot because the simulator does not match reality: friction, mass, latency, and camera noise all differ. This mismatch is called the reality gap.

Domain randomisation trains the policy over a wide distribution of simulator parameters, such as object mass, friction coefficients, lighting, and textures, so the real world looks like one more sample. System identification takes the opposite approach and fits the simulator parameters to measurements from the real system.

Fine-tuning on a small amount of real data after simulation training closes the remaining gap. Evaluations report success rate in simulation and on hardware, and the drop between them.

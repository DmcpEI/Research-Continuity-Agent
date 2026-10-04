# Deformable Bag Manipulation

Plastic and fabric bags are deformable: their shape changes with every contact, so a fixed grasp pose rarely works twice. Bag manipulation tasks include opening the bag, keeping it open, inserting items, and lifting the bag by its handles.

A typical pipeline first detects the bag rim and handles in an RGB image, then uses a bimanual action to pull the rim apart and enlarge the opening. A learned policy estimates the opening area from a top-down camera and repeats the dilation action until the opening is large enough for insertion.

Inserting an item requires the bag to stay open while one arm releases it. Failure modes include the bag collapsing, items sliding off the rim, and handles twisting when the bag is lifted.

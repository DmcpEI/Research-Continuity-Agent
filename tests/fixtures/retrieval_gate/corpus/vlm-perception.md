# Vision-Language Models for Grocery Perception

Grocery packing robots must recognise each item and decide how to handle it: fragile items such as eggs go on top, heavy items such as bottles go at the bottom.

Two perception pipelines were compared. Pipeline A runs a YOLO object detector trained on grocery classes and looks up handling properties in a table. Pipeline B passes the YOLO crops to a vision-language model (VLM) that answers questions about fragility, weight class, and deformability directly from the image.

Pipeline B generalised to unseen products because the VLM reasons from appearance, while Pipeline A failed on any class missing from its training set. The cost is latency: the VLM adds roughly a second per item on a laptop GPU. Structured prompts that force a fixed answer format reduced invalid VLM outputs.

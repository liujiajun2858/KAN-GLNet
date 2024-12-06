# KGL-Net
We recommend running it on the Ubuntu 22.04 system with a 3090 GPU and CUDA 11.8.<br>
**block**
The **block** contains GLFN feature modulation blocks, ContraNorm, and Reverse Bottleneck KAN Convolutions, with the Reverse Bottleneck KAN Convolutions located in the `fastkan.py` file.<br>

**Train**<br>
`python train_semseg.py`<br>
**Test**<br>
`python test_semseg.py`<br>

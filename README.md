# KAN-GLNet
We recommend running it on the Ubuntu 22.04 system with a 3090 GPU and CUDA 11.8.<br>
**block**<br>
The **block** contains GLFN feature modulation blocks, ContraNorm, and Reverse Bottleneck KAN Convolutions, with the Reverse Bottleneck KAN Convolutions located in the `rbkan.py` file.<br>
**Data**<br>
The corresponding rapeseed point cloud dataset can be obtained at the following link: (https://pan.baidu.com/s/1cSPUx2l-cW-66iBHUaF2Aw?pwd=vtx3#list/path=%2FData). After downloading and extracting, simply place it directly in the `data` directory of the model.<br>
**Train**<br>
`python train.py`<br>
**Test**<br>
`python test_semseg.py`<br>
**Log**<br>
The complete model weights along with training and testing printouts are saved in the log folder. Additional information can be obtained at the following link: (https://pan.baidu.com/s/1j6EzeyflBjbOCfjotgDiwQ?pwd=4gd2).
**This code repository will be continuously improved and updated.**


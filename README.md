# KAN-GLNet
We recommend running it on the Ubuntu 22.04 system with a 3090 GPU and CUDA 11.8.<br>
**block**<br>
The **block** contains GLFN feature modulation blocks, ContraNorm, and Reverse Bottleneck KAN Convolutions, with the Reverse Bottleneck KAN Convolutions located in the `rbkan.py` file.<br>
**Data**<br>
The corresponding rapeseed point cloud dataset can be obtained at the following link: (https://pan.baidu.com/s/1C7bPSdd9UA-xx3EcqYC7Yg?pwd=fajs). After downloading and extracting, simply place it directly in the `data` directory of the model.<br>
**log**<br>
The **log** file will generate the corresponding results and weight information. For visualizing the results, it is recommended to use software such as CloudCompare or MeshLab.<br>
**Train**<br>
`python train_semseg.py`<br>
**Test**<br>
`python test_semseg.py`<br>
**This code repository will be continuously improved and updated.**


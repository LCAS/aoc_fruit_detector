# AOC Fruit Detector using Detectron2 MaskRCNN

Instance segmentation of a scene and output Mask-RCNN predictions as images and json message/file (Agri-OpenCore)

![Example images](./src/aoc_fruit_detector/scripts/data/figure/output_fig.png)

## Installation and Requirements

Install following required packages or check/install required versions from requirements.txt file

`python3` `torchvision` `pickle` `numpy` `opencv-python` `scikit-image` `matplotlib`
`detectron2`

```
pip install -r requirements. txt
```
Clone Detectron2 package from GitHub and install the package into your workspace.

```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

## Usage

Run package to publish annotations detected by aoc_fruit_detector package 

```bash
ros2 launch aoc_fruit_detector fruit_detection.launch.py 
```


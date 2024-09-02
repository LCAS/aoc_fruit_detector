# Fruit Detector for Detectron2 MaskRCNN

Instance segmentation of a scene and output Mask-RCNN predictions as images and json message/file (Agri-OpenCore)

![Example images](./data/figure/output_fig.png)

## Requirements
`python3` `torchvision` `pickle` `numpy` `opencv-python` `scikit-image` `matplotlib`
`detectron2`

## Installation

```
https://github.com/usmanzahidi/FruitDetector.git
```

## Usage

```bash
usage: main.py [-O]
-O  for optimized model (alphabet O)
without -O the script is executed in Debug mode and outputs json files in './annotations/predicted' and prediction images in '.output/predicted_images' folders

The configuration settings are read from config.yaml

```

## Example:

```bash
python -O main.py
OR
python main.py
```

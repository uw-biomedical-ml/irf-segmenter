# irf-segmenter

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/uw-biomedical-ml/irf-segmenter/blob/master/LICENSE)

## Fully-automated, deep learning based, intraretinal fluid segmenter for optical coherence (OCT) images of the macula

## Usage

```
usage: run.py [-h] [--mode MODE] input_file output_file

positional arguments:
  input_file   Input PNG file
  output_file  Output PNG file

optional arguments:
  -h, --help   show this help message and exit
  --mode MODE  Output mode, 'mask_blend' (default) for masked heatmap output,
               'mask' for binary mask output, 'blend' for blended heatmap
```

## Example

Input image:

![input image example](https://github.com/uw-biomedical-ml/irf-segmenter/raw/master/example.png "example.png")

Mask Blended image:

![input image example](https://github.com/uw-biomedical-ml/irf-segmenter/raw/master/output-mask_blend.png "output-mask_blend.png")

Blended image:
![input image example](https://github.com/uw-biomedical-ml/irf-segmenter/raw/master/output-blend.png "output-blend.png")

Binary Mask image:
![input image example](https://github.com/uw-biomedical-ml/irf-segmenter/raw/master/output-mask.png "output-mask.png")

## Requirements

You must install the following Python packages:

- TensorFlow 
  - [See installation instructions](https://www.tensorflow.org/install/).
- Keras
  - [See installation instructions](https://github.com/fchollet/keras).
- HDF5 and h5py
- Pillow
- Matplotlib
- cv2

`conda install --file requirements.txt`

or

`pip install -r requirements.txt`

If having troubles with `import cv2`, try:

`conda install --channel https://conda.anaconda.org/menpo opencv3`

## Installation

`git clone https://github.com/uw-biomedical-ml/irf-segmenter`





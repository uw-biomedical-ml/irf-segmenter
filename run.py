#!/usr/bin/env python
# Copyright Aaron Lee, University of Washington 2017
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Lambda

from keras.models import load_model
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="mask_blend", type=str, help="Output mode, 'mask_blend' (default) for masked heatmap output, 'mask' for binary mask output, 'blend' for blended heatmap" )
parser.add_argument('input_file', help='Input PNG file')
parser.add_argument('output_file', help='Output PNG file')
args = parser.parse_args()

if not (args.mode == "mask_blend" or args.mode == "blend" or args.mode == "mask"):
  print("Invalid mode: %s" % args.mode)
  sys.exit()

modelfile = 'weights.hdf5'
image_rows = 432
image_cols = 32


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [ shape[:1] // parts, shape[1:] ])
        stride = tf.concat(0, [ shape[:1] // parts, shape[1:]*0 ])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    with tf.device('/cpu:0'):
        return Model(input=model.inputs, output=outputs)

def get_unet():
	    inputs = Input((1, image_rows, image_cols))
	    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
	    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
	    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
	    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
	    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
	    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
	    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
	    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
	    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
	    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)


	    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
	    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

	    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
	    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

	    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
	    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

	    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
	    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

	    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

	    model = Model(input=inputs, output=conv10)

	    return model

model = get_unet()
model = make_parallel(model, 1)
model.load_weights(modelfile)

params = []
with open("params.txt") as fin:
  for l in fin:
    arr = l.rstrip().split("\t")
    params.append(np.float32(arr[1]))

pngfile = args.input_file
img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE)
img = np.array([img])
imgori = np.copy(img)
imgori = imgori.reshape((img.shape[1], img.shape[2]))
ji = Image.fromarray(imgori)
img = img.astype('float32')

img -= params[0] # subtract by mean, divide by SD
img /= params[1]

totaloutput = np.zeros((img.shape[1], img.shape[2], 32))
for dx in range(0, img.shape[2] - 32):
    imgs = img[0, 0:image_rows, dx:image_cols+dx]
    imgs = imgs.reshape((  1, image_rows,image_cols))
    imgsbatch = np.zeros((1, 1, image_rows,image_cols))
    imgsbatch[0] = imgs

    output = model.predict(imgsbatch, batch_size=1) # inference step
    for x in range(0, image_rows):
        for y in range(0, image_cols):
            totaloutput[x,dx + y,dx % 32] = output[0,0,x,y]

totaloutput = np.mean(totaloutput, 2)

if (args.mode == "mask"):
    # for binary masks
    mask = (totaloutput > 0.5)
    mask = np.uint8(mask)
    mask *= 255
    mask = Image.fromarray(mask)
    mask.save(args.output_file)
elif (args.mode == "mask_blend"):
    # for masked heatmap overlay
    mask = (totaloutput < 0.5)
    mask = np.uint8(mask)
    mask *= 255
    mask = Image.fromarray(mask)
    my_cm = matplotlib.cm.get_cmap('jet')
    mapped_data = my_cm(totaloutput, bytes=True)
    j = Image.fromarray(mapped_data).convert('RGBA')
    ji = ji.convert("RGBA")
    Image.composite(ji, j,mask).save(args.output_file)
elif (args.mode == "blend"):
    # for blend overlay
    my_cm = matplotlib.cm.get_cmap('jet')
    mapped_data = my_cm(totaloutput, bytes=True)
    j = Image.fromarray(mapped_data).convert('RGBA')
    ji = ji.convert("RGBA")
    Image.blend(ji, j,0.2).save(args.output_file)

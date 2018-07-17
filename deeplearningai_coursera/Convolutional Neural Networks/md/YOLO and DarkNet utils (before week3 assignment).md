
### import dependencies


```python
import colorsys
import imghdr
import os
import random
from keras import backend as K

import numpy as np
from PIL import Image, ImageDraw, ImageFont
```

    Using TensorFlow backend.



```python
!wget https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/coco_classes.txt
!wget https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/object_classes.txt
!wget https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/view/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo.h5
!wget https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo_anchors.txt
```

    --2018-07-16 18:43:44--  https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/coco_classes.txt
    Resolving hub.coursera-notebooks.org (hub.coursera-notebooks.org)... 34.192.29.60, 52.205.26.206
    Connecting to hub.coursera-notebooks.org (hub.coursera-notebooks.org)|34.192.29.60|:443... connected.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/coco_classes.txt [following]
    --2018-07-16 18:43:44--  https://hub.coursera-notebooks.org/hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/coco_classes.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fcoco_classes.txt [following]
    --2018-07-16 18:43:44--  https://hub.coursera-notebooks.org/hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fcoco_classes.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 2603 (2.5K) [text/html]
    Saving to: ‘coco_classes.txt’
    
    coco_classes.txt    100%[===================>]   2.54K  --.-KB/s    in 0s      
    
    2018-07-16 18:43:45 (97.9 MB/s) - ‘coco_classes.txt’ saved [2603/2603]
    
    --2018-07-16 18:43:46--  https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/object_classes.txt
    Resolving hub.coursera-notebooks.org (hub.coursera-notebooks.org)... 34.192.29.60, 52.205.26.206
    Connecting to hub.coursera-notebooks.org (hub.coursera-notebooks.org)|34.192.29.60|:443... connected.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/object_classes.txt [following]
    --2018-07-16 18:43:46--  https://hub.coursera-notebooks.org/hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/object_classes.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fobject_classes.txt [following]
    --2018-07-16 18:43:46--  https://hub.coursera-notebooks.org/hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fobject_classes.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 2603 (2.5K) [text/html]
    Saving to: ‘object_classes.txt’
    
    object_classes.txt  100%[===================>]   2.54K  --.-KB/s    in 0s      
    
    2018-07-16 18:43:46 (87.8 MB/s) - ‘object_classes.txt’ saved [2603/2603]
    
    --2018-07-16 18:43:47--  https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/view/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo.h5
    Resolving hub.coursera-notebooks.org (hub.coursera-notebooks.org)... 34.192.29.60, 52.205.26.206
    Connecting to hub.coursera-notebooks.org (hub.coursera-notebooks.org)|34.192.29.60|:443... connected.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/user/lzstqjudclarqyimsfzdhu/view/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo.h5 [following]
    --2018-07-16 18:43:48--  https://hub.coursera-notebooks.org/hub/user/lzstqjudclarqyimsfzdhu/view/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo.h5
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fview%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fyolo.h5 [following]
    --2018-07-16 18:43:48--  https://hub.coursera-notebooks.org/hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fview%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fyolo.h5
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 2603 (2.5K) [text/html]
    Saving to: ‘yolo.h5’
    
    yolo.h5             100%[===================>]   2.54K  --.-KB/s    in 0s      
    
    2018-07-16 18:43:48 (10.5 MB/s) - ‘yolo.h5’ saved [2603/2603]
    
    --2018-07-16 18:43:49--  https://hub.coursera-notebooks.org/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo_anchors.txt
    Resolving hub.coursera-notebooks.org (hub.coursera-notebooks.org)... 34.192.29.60, 52.205.26.206
    Connecting to hub.coursera-notebooks.org (hub.coursera-notebooks.org)|34.192.29.60|:443... connected.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo_anchors.txt [following]
    --2018-07-16 18:43:49--  https://hub.coursera-notebooks.org/hub/user/lzstqjudclarqyimsfzdhu/edit/week3/Car%20detection%20for%20Autonomous%20Driving/model_data/yolo_anchors.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 302 Moved Temporarily
    Location: /hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fyolo_anchors.txt [following]
    --2018-07-16 18:43:49--  https://hub.coursera-notebooks.org/hub/login?next=%2Fhub%2Fuser%2Flzstqjudclarqyimsfzdhu%2Fedit%2Fweek3%2FCar%2520detection%2520for%2520Autonomous%2520Driving%2Fmodel_data%2Fyolo_anchors.txt
    Reusing existing connection to hub.coursera-notebooks.org:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 2603 (2.5K) [text/html]
    Saving to: ‘yolo_anchors.txt’
    
    yolo_anchors.txt    100%[===================>]   2.54K  --.-KB/s    in 0s      
    
    2018-07-16 18:43:50 (107 MB/s) - ‘yolo_anchors.txt’ saved [2603/2603]
    



```python
ls
```

    coco_classes.txt  FiraMono-Medium.otf  [0m[01;34mMask_RCNN[0m/  [01;34msamples[0m/          yolo.h5
    [01;34mdatalab[0m/          [01;34mimages[0m/              [01;34mmrcnn[0m/      yolo_anchors.txt



```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```



     <input type="file" id="files-bfa39353-2a24-47de-aa23-8a57baf5376d" name="files[]" multiple disabled />
     <output id="result-bfa39353-2a24-47de-aa23-8a57baf5376d">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving FiraMono-Medium.otf to FiraMono-Medium.otf
    User uploaded file "FiraMono-Medium.otf" with length 127344 bytes


### yolo_utils


```python
#@title Default title text
def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
        # print(class_names)
    class_names = [c.strip() for c in class_names]
    return class_names
class_names = read_classes('coco_classes.txt')
```


```python
def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2) # n rows, 2 cols
    return anchors

read_anchors('yolo_anchors.txt')
```




    array([[0.57273 , 0.677385],
           [1.87446 , 2.06253 ],
           [3.33843 , 5.47434 ],
           [7.88282 , 3.52778 ],
           [9.77052 , 9.16828 ]])




```python
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors
```


```python
generate_colors(class_names)
```




    [(255, 38, 0),
     (0, 63, 255),
     (255, 114, 0),
     (127, 255, 0),
     (146, 255, 0),
     (223, 255, 0),
     (0, 255, 25),
     (242, 0, 255),
     (0, 235, 255),
     (0, 140, 255),
     (0, 44, 255),
     (255, 191, 0),
     (0, 255, 6),
     (255, 19, 0),
     (255, 76, 0),
     (70, 255, 0),
     (0, 159, 255),
     (255, 248, 0),
     (255, 229, 0),
     (0, 255, 159),
     (255, 133, 0),
     (255, 172, 0),
     (0, 197, 255),
     (12, 255, 0),
     (255, 0, 95),
     (0, 255, 255),
     (89, 255, 0),
     (0, 6, 255),
     (242, 255, 0),
     (146, 0, 255),
     (0, 255, 44),
     (0, 216, 255),
     (70, 0, 255),
     (255, 0, 57),
     (184, 255, 0),
     (0, 82, 255),
     (184, 0, 255),
     (255, 0, 152),
     (0, 25, 255),
     (255, 0, 19),
     (0, 255, 178),
     (255, 0, 191),
     (0, 255, 197),
     (255, 210, 0),
     (255, 0, 76),
     (204, 0, 255),
     (255, 153, 0),
     (0, 255, 82),
     (255, 95, 0),
     (165, 0, 255),
     (51, 255, 0),
     (127, 0, 255),
     (0, 255, 140),
     (255, 0, 229),
     (255, 0, 38),
     (0, 121, 255),
     (255, 0, 114),
     (12, 0, 255),
     (203, 255, 0),
     (255, 0, 210),
     (108, 0, 255),
     (50, 0, 255),
     (89, 0, 255),
     (255, 57, 0),
     (255, 0, 0),
     (31, 0, 255),
     (255, 0, 172),
     (0, 255, 121),
     (0, 178, 255),
     (31, 255, 0),
     (255, 0, 248),
     (108, 255, 0),
     (223, 0, 255),
     (165, 255, 0),
     (0, 255, 102),
     (0, 102, 255),
     (0, 255, 216),
     (0, 255, 235),
     (255, 0, 133),
     (0, 255, 63)]




```python
def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    print(image_dims.shape)
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes
```


```python
def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data
```


```python
def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
```


```python
[i for i in reversed([1,2,3,4])]
```




    [4, 3, 2, 1]



## YOLO_v2 Model Defined in Keras.

### utils

复合函数


```python
from functools import reduce


def compose(*funcs):
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
```

### Yad2k: Yet Another DarkNet to Keras


```python
"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def darknet_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512))


def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    body = darknet_body()(inputs)
    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
    return Model(inputs, logits)

```

### Another easier way for DarkNet (using Tensorflow)


```python
import tensorflow as tf
```


```python
def conv_block(net, kernel, bias, name = 'None'):
  conv = tf.nn.conv2d(net, kernel, strides=[1,1,1,1], padding='SAME')
  conv = tf.nn.bias_add(conv, bias)
  conv = tf.nn.leaky_relu(conv)
  return conv
```


```python
conv = conv_block(net, kernel1, bias1)
conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
conv = conv_block(conv, kernel2, bias2)
conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
conv = conv_block(conv, kernel3, bias3)
conv = conv_block(conv, kernel4, bias4)
conv = conv_block(conv, kernel5, bias5)
conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
conv = conv_block(conv, kernel6, bias6)
conv = conv_block(conv, kernel7, bias7)
conv = conv_block(conv, kernel8, bias8)
conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
conv = conv_block(conv, kernel9, bias9)
conv = conv_block(conv, kernel10, bias10)
conv = conv_block(conv, kernel11, bias11)
conv = conv_block(conv, kernel12, bias12)
conv = conv_block(conv, kernel13, bias13)
conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
conv = conv_block(conv, kernel14, bias14)
conv = conv_block(conv, kernel15, bias15)
conv = conv_block(conv, kernel16, bias16)
conv = conv_block(conv, kernel17, bias17)
conv = conv_block(conv, kernel18, bias18)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-1f6e16b32882> in <module>()
    ----> 1 conv = conv_block(net, kernel1, bias1)
          2 conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
          3 conv = conv_block(conv, kernel2, bias2)
          4 conv = tf.contrib.layers.max_pool2d(conv, [2,2], stride=2)
          5 conv = conv_block(conv, kernel3, bias3)


    NameError: name 'net' is not defined



```python
if imagenet_classify:
  conv = conv_block(conv, kernel19, bias19, name='None')
  conv = tf.nn.avg_pool(conv, 
                        ksize=[1, conv.get_shape()[1], conv.get_shape()[2],1],
                        strides=[1, conv.get_shape()[1], conv.get_shape()[2],1],
                        padding='VALID')
  ## flatten
  conv = tf.reshape(conv, shape=[-1])
  conv = tf.nn.softmax(conv)

conv
```

### keras yolo


```python
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.models import Model


sys.path.append('..')

voc_anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
```


```python
def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V2 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body()(inputs))
    conv20 = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(1024, (3, 3)))(darknet.output)

    conv13 = darknet.layers[43].output
    conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    conv21_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv21)

    x = concatenate([conv21_reshaped, conv20])
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    return Model(inputs, x)
```


```python
def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    
    # tile(x, n), 将x在各个维度上重复n次，x为张量，n为与x维度数目相同的列表
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))
    
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))


    box_confidence = K.sigmoid(feats[..., 4:5])
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_confidence, box_xy, box_wh, box_class_probs
```


```python
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])
```


```python
def yolo_loss(args,
              anchors,
              num_classes,
              rescore_confidence=False,
              print_loss=False):
    """YOLO localization loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    if print_loss:
        total_loss = tf.Print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss

```


```python
def yolo(inputs, anchors, num_classes):
    """Generate a complete YOLO_v2 localization model."""
    num_anchors = len(anchors)
    body = yolo_body(inputs, num_anchors, num_classes)
    outputs = yolo_head(body.output, anchors, num_classes)
    return outputs
```


```python

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    return boxes, scores, classes
```


```python
def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    ## add masks
    boxes, scores, classes = yolo_filter_boxes(
        box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    
    return boxes, scores, classes
```


```python

def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = min(np.floor(box[0]).astype('int'),1)
        best_iou = 0
        best_anchor = 0
                
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k
                
        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]),
                    np.log(box[3] / anchors[best_anchor][1]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes
```

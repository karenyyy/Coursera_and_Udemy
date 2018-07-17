
## YOLO 试运行


```python
!git clone https://github.com/hizhangp/yolo_tensorflow.git
```

    Cloning into 'yolo_tensorflow'...
    remote: Counting objects: 117, done.[K
    remote: Total 117 (delta 0), reused 0 (delta 0), pack-reused 117[K
    Receiving objects: 100% (117/117), 198.73 KiB | 11.69 MiB/s, done.
    Resolving deltas: 100% (61/61), done.



```python
!./yolo_tensorflow/download_data.sh
```

    Creating data directory...
    Downloading Pascal VOC 2012 data...
    --2018-07-17 02:13:48--  http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    Resolving pjreddie.com (pjreddie.com)... 128.208.3.39
    Connecting to pjreddie.com (pjreddie.com)|128.208.3.39|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar [following]
    --2018-07-17 02:13:48--  https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    Connecting to pjreddie.com (pjreddie.com)|128.208.3.39|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 460032000 (439M) [application/octet-stream]
    Saving to: ‘VOCtrainval_06-Nov-2007.tar’
    
    VOCtrainval_06-Nov- 100%[===================>] 438.72M  44.2MB/s    in 15s     
    
    2018-07-17 02:14:03 (29.2 MB/s) - ‘VOCtrainval_06-Nov-2007.tar’ saved [460032000/460032000]
    
    Extracting VOC data...
    Done.



```python
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```



     <input type="file" id="files-8f8474ef-c627-4fdd-b00e-c573b42300bf" name="files[]" multiple disabled />
     <output id="result-8f8474ef-c627-4fdd-b00e-c573b42300bf">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 



```python
!mkdir weights
!mv YOLO_small.tar.gz ./weights
!mkdir data
!mv weights ./data

```

    mv: cannot stat 'YOLO_small.tar.gz': No such file or directory
    mkdir: cannot create directory ‘data’: File exists



```python
ls data/weights/
```


```python
!python yolo_tensorflow/train.py
```

    WARNING:tensorflow:From /content/yolo_tensorflow/yolo/yolo_net.py:186: calling reduce_max (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    Processing gt_labels from: data/pascal_voc/VOCdevkit/VOC2007
    Saving gt_labels to: data/pascal_voc/cache/pascal_train_gt_labels.pkl
    Appending horizontally-flipped training examples ...
    2018-07-17 02:15:22.433503: W tensorflow/core/framework/allocator.cc:108] Allocation of 24084480 exceeds 10% of system memory.
    2018-07-17 02:15:22.757500: W tensorflow/core/framework/allocator.cc:108] Allocation of 102760448 exceeds 10% of system memory.
    2018-07-17 02:15:22.767540: W tensorflow/core/framework/allocator.cc:108] Allocation of 18874368 exceeds 10% of system memory.
    2018-07-17 02:15:23.521142: W tensorflow/core/framework/allocator.cc:108] Allocation of 18874368 exceeds 10% of system memory.
    2018-07-17 02:15:23.672302: W tensorflow/core/framework/allocator.cc:108] Allocation of 37748736 exceeds 10% of system memory.
    Start training ...
    tcmalloc: large alloc 1327718400 bytes == 0x6d4a6000 @  0x7f39eb149107 0x7f39de34d575 0x7f39e04ddc19 0x7f39e04e823a 0x7f39e04e8a1d 0x7f39e0506d87 0x7f39e05071e9 0x7f39e0508525 0x7f39dc564c5c 0x7f39dc51c5fd 0x7f39dc50e2a5 0x7f39dc57b7c2 0x7f39dc579707 0x7f39e9a8c0ff 0x7f39eaefe7fc 0x7f39ea085b5f (nil)
    tcmalloc: large alloc 1301012480 bytes == 0xcf5a4000 @  0x7f39eb149107 0x7f39de34d575 0x7f39e04ddc19 0x7f39e04e823a 0x7f39e04e8a1d 0x7f39e0506d87 0x7f39e05071e9 0x7f39e0508525 0x7f39dc564c5c 0x7f39dc51c5fd 0x7f39dc50e2a5 0x7f39dc57b7c2 0x7f39dc579707 0x7f39e9a8c0ff 0x7f39eaefe7fc 0x7f39ea085b5f (nil)



```python
!python yolo_tensorflow/test.py
```

## YOLONET 源码阅读

与大部分目标检测与识别方法（比如Fast R-CNN）将目标识别任务分类多个流程不同:
- 目标区域预测
- 类别预测

YOLO将目标区域预测和目标类别预测整合于__单个神经网络模型__中，实现在准确率较高的情况下快速目标检测与识别，更加适合现场应用环境。

网络建立是通过build_networks()方法实现的，网络由卷积层-pooling层和全连接层组成

__网络接受输入维度为([None, 448, 448, 3])，输出维度为([None,1470]__


![](https://pic4.zhimg.com/80/v2-ee4db90336d60d251d7254f9918c3a48_hd.jpg)



__loss函数代码的关键，loss函数定义为__

![](https://pic3.zhimg.com/80/v2-99be5fd97cee75068fbbe82f8c381275_hd.jpg)






```python
import numpy as np
import tensorflow as tf
import yolo_tensorflow.yolo.config as cfg

slim = tf.contrib.slim
```


```python
class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes) #类别数量，值为20
        self.image_size = cfg.IMAGE_SIZE  #图像尺寸,值为448
        self.cell_size = cfg.CELL_SIZE   #cell尺寸，值为7
        self.boxes_per_cell = cfg.BOXES_PER_CELL #每个grid cell负责的boxes，默认为2
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)  #输出尺寸
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE #学习速率LEARNING_RATE = 0.0001
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        #将网络输出分离为类别和定位以及box大小，输出维度为7*7*20+7*7*2+7*7*2*4=1470
        with tf.variable_scope(scope):
            #类别，shape为(45, 7, 7, 20)
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            #定位，shape为(45, 7, 7, 2)
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            ##box大小，长宽等 shape为(45, 7, 7, 2, 4)
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            #label的类别结果，shape为(45, 7, 7, 1)
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            #label的定位结果，shape为(45, 7, 7, 1, 4) => [x,y,w,h]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
```

## 训练


```python
class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS  #速率延迟步数DECAY_STEPS = 30000
        self.decay_rate = cfg.DECAY_RATE #延迟率DECAY_RATE = 0.1
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER 
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},'''
                    ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''
                    '''' Load: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

```

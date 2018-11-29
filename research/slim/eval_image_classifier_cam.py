# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import os
from grad_cam_plus_plus import GradCamPlusPlus
import numpy as np
import cv2

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 1,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'D:\pretrained',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', './eval_result', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'recall_at_k', 5,
    'recall at k')

tf.app.flags.DEFINE_string(
    'dataset_name', 'common', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', "D:\data\\fashion\\fashion_style14_v1\FashionStyle14_v1\\tfrecord-cls",
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_string('last_conv_layer', "Conv2d_7b_1x1", 'last_conv_layer')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits_op, end_points = network_fn(images)

        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits_op, 1)
        labels = tf.squeeze(labels)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

        cam = GradCamPlusPlus(logits_op, end_points[FLAGS.last_conv_layer], images)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=variables_to_restore)
        saver.restore(sess, checkpoint_path)
        xs = None
        ys = None
        logits = None
        for i in range(num_batches):
            print(i, num_batches)
            result = sess.run([images, labels, logits_op, accuracy_op])
            if xs is None:
                xs = result[0]
                ys = result[1]
                logits = result[2]
            else:
                xs = np.concatenate((xs, result[0]), axis=0)
                ys = np.concatenate((ys, result[1]), axis=0)
                logits = np.concatenate((logits, result[2]), axis=0)
            print(result[3])
        cam_imgs, class_indices = cam.create_cam_imgs(sess, xs, logits)
        heatmap_imgs = {}
        for i in range(len(xs)):
            heatmap = cam.convert_cam_2_heatmap(cam_imgs[i][0])
            overlay_img = cam.overlay_heatmap(xs[i], heatmap)

            if ys[i].argmax() == logits[i].argmax():
                key = "true/label_%d" % ys[i].argmax()
            else:
                key = "false/truth_%d_pred_%d" % (ys[i].argmax(), logits[i].argmax())
            if key not in heatmap_imgs:
                heatmap_imgs[key] = []
            if len(xs[i].shape) != 3 or xs[i].shape[2] != 3:
                img = cv2.cvtColor(xs[i], cv2.COLOR_GRAY2BGR)[..., ::-1]
            else:
                img = xs[i]
            heatmap_imgs[key].append(img)
            heatmap_imgs[key].append(overlay_img[..., ::-1])
        writer = tf.summary.FileWriter(FLAGS.eval_dir)
        for key in heatmap_imgs:
            cam.write_summary(writer, "grad_cam_%s" % key, heatmap_imgs[key], sess)


if __name__ == '__main__':
    tf.app.run()

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
import os, glob
from grad_cam_plus_plus import GradCamPlusPlus
import embedding_visualizer as embedding
import numpy as np
import cv2
import random

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
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
    'preprocessing_name', "inception", 'The name of the preprocessing to use. If left '
                                       'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_string('last_conv_layer', "Conv2d_7b_1x1", 'last_conv_layer')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 299, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')
tf.app.flags.DEFINE_bool(
    'show_original_image', False, 'show_original_image')

tf.app.flags.DEFINE_bool(
    'use_cam', True, 'use_cam')

tf.app.flags.DEFINE_bool(
    'use_embedding', True, 'use_embedding')
tf.app.flags.DEFINE_integer(
    'num_embedding', 200, 'num_embedding')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset_slim = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset_slim.num_classes - FLAGS.labels_offset),
        is_training=False)

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    def train_pre_process(example_proto):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    "image/class/name": tf.FixedLenFeature((), tf.string, default_value=""),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], 3)

        if image_preprocessing_fn is not None:
            image = image_preprocessing_fn(image, FLAGS.eval_image_size, FLAGS.eval_image_size)
        else:
            image = tf.cast(image, tf.float32)

            image = tf.expand_dims(image, 0)
            image = tf.image.resize_image_with_pad(image, FLAGS.eval_image_size, FLAGS.eval_image_size)
            # image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
            image = tf.squeeze(image, [0])

            image = tf.divide(image, 255.0)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)

        label = parsed_features["image/class/label"]
        label_name = parsed_features["image/class/name"]
        return image, label, label_name

    files_op = tf.placeholder(tf.string, shape=[None], name="files")
    dataset = tf.data.TFRecordDataset(files_op)
    dataset = dataset.map(train_pre_process)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()
    images, labels, label_names = iterator.get_next()

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
    label_map = {0: "conservative",
                 1: "dressy",
                 2: "ethnic",
                 3: "fairy",
                 4: "feminine",
                 5: "gal",
                 6: "girlish",
                 7: "kireime-casual",
                 8: "lolita",
                 9: "mode",
                 10: "natural",
                 11: "retro",
                 12: "rock",
                 13: "street"}
    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = int(math.ceil(dataset_slim.num_samples / float(FLAGS.batch_size)))

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
    ys_names = None
    tf_record_files = glob.glob(os.path.join(FLAGS.dataset_dir, "*%s*tfrecord" % FLAGS.dataset_split_name))
    sess.run(iterator.initializer, feed_dict={files_op: tf_record_files})
    total_accuracies = 0.
    for i in range(num_batches):
        print("step %d/%d" % (i + 1, num_batches))
        result = sess.run([images, labels, logits_op, accuracy_op, label_names])
        if xs is None:
            xs = result[0]
            ys = result[1]
            logits = result[2]
            ys_names = result[4]
        else:
            xs = np.concatenate((xs, result[0]), axis=0)
            ys = np.concatenate((ys, result[1]), axis=0)
            logits = np.concatenate((logits, result[2]), axis=0)
            ys_names = np.concatenate((ys_names, result[4]), axis=0)
        total_accuracies += result[3]
        print(result[3])
    print("accuracy", total_accuracies / num_batches)
    if FLAGS.use_cam:
        cam_imgs, class_indices = cam.create_cam_imgs(sess, xs, logits)
        heatmap_imgs = {}

        for i in range(len(xs)):
            heatmap = cam.convert_cam_2_heatmap(cam_imgs[i][0])
            overlay_img = cam.overlay_heatmap(xs[i], heatmap)

            if ys[i] == logits[i].argmax():
                key = "true/label_%s-%d" % (ys_names[i], ys[i])
            else:
                key = "false/truth_%s-%d_pred_%s-%d" % (ys_names[i], ys[i], label_map[logits[i].argmax()],
                                                        logits[i].argmax())
            if key not in heatmap_imgs:
                heatmap_imgs[key] = []
            if len(xs[i].shape) != 3 or xs[i].shape[2] != 3:
                img = cv2.cvtColor(xs[i], cv2.COLOR_GRAY2BGR)[..., ::-1]
            else:
                img = xs[i]
            if FLAGS.show_original_image:
                heatmap_imgs[key].append(img)
            heatmap_imgs[key].append(overlay_img[..., ::-1])
        writer = tf.summary.FileWriter(FLAGS.eval_dir)
        for key in heatmap_imgs:
            cam.write_summary(writer, "grad_cam_%s" % key, heatmap_imgs[key], sess, FLAGS.eval_image_size)
        print("finished to summary cam")
    if FLAGS.use_embedding:
        random.seed(1)
        indices = list(range(len(xs)))
        random.shuffle(indices)
        embedding.summary_embedding(sess, xs[indices[:FLAGS.num_embedding]], [logits[indices[:FLAGS.num_embedding]]],
                                    os.path.join(os.path.abspath(FLAGS.eval_dir), "embedding"), FLAGS.eval_image_size,
                                    channel=3, labels=ys[indices[:FLAGS.num_embedding]],
                                    prefix="eval_embedding", label_map=label_map)
        print("finished to summary embedding")
    print("finished to evaluate")


if __name__ == '__main__':
    tf.app.run()

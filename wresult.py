# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple speech recognition to spot a limited number of keywords.

re:
This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

me:I have save the result, and try to test it using test data, generate the accuracy metrics, save in the expected format.




It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train


me: To run the test process, use:
bazel run tensorflow/examples/speech_commands:ytest



This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

me: using the checkout, also try to learning to run the freeze script  -- what is binary GraphDef
and how to deploy on mobile applications






If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile


import os # os.path -- common pathname manipulations
import csv
import random



from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util





import glob
from tensorflow.python.ops import io_ops
import numpy as np


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture):
  """Creates an audio model with the nodes needed for inference.                     here the wav_data_placeholder and logit may be the input and out nodes, But can they be used as global variable, or I must return them

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    clip_stride_ms: How often to run recognition. Useful for models with cache.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
  """

  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  wav_loader = io_ops.read_file(wav_data_placeholder)
  decoded_sample_data = contrib_audio.decode_wav(
      wav_loader,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  fingerprint_input = contrib_audio.mfcc(
      spectrogram,
      decoded_sample_data.sample_rate,
      dct_coefficient_count=dct_coefficient_count)
  fingerprint_frequency_size = model_settings['dct_coefficient_count']
  fingerprint_time_size = model_settings['spectrogram_length']
  reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
  ])

  logits = models.create_model(
      reshaped_input, model_settings, model_architecture, is_training=False,
      runtime_settings=runtime_settings)

  # Create an output to use for inference.
  output=tf.nn.softmax(logits, name='labels_softmax')
  return (wav_data_placeholder, logits, output)

FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()


  # the placepolder, also have a name 'wav_data', the output also has the name 'labels_softmax'
  # here load our graph, but the freeze, the problem is the can not find the files
  # ANS: maybe it is because I don't give the path of the start checkpoint .  search: load_variables_from checkpoint
  wav_data_placeholder, logits, softmaxoutput=create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate,
                         FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
                         FLAGS.window_size_ms, FLAGS.window_stride_ms,
                         FLAGS.dct_coefficient_count, FLAGS.model_architecture)
  models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

  # Turn all the variables into inline constants inside the graph and save it.
  # frozen_graph_def = graph_util.convert_variables_to_constants(
  #     sess, sess.graph_def, ['labels_softmax'])
  '''
  tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(FLAGS.output_file),
        os.path.basename(FLAGS.output_file),
        as_text=False)
  tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)
  '''

  #output=tf.get_default_graph().get_operation_by_name('labels_softmax')
  #wav_file=tf.get_default_graph().get_operation_by_name('wav_data')
  # output=frozen_graph_def.get_operation_by_name('labels_softmax')
  # wav_file=frozen_graph_def.get_operation_by_name('wav_data')


  SILENCE_LABEL = 'silence'
  SILENCE_INDEX = 0
  UNKNOWN_WORD_LABEL = 'unknown'
  UNKNOWN_WORD_INDEX = 1
  lableList= [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + FLAGS.wanted_words.split(',')
  print("lableList:%s",lableList)


  if not os.path.exists(FLAGS.result_dir):
    os.makedirs(FLAGS.result_dir)
  with open(FLAGS.result_dir + '/logit_p6_12_submission.csv',
          'w') as csvfile:  # python input and output  Reading and Writing Files  2.7 , newline='' can not be used
    # 'w' for only writing (an existing file with the same name will be erased), and 'a' opens the file for appending;
    randwriter = csv.writer(csvfile,
                            delimiter=',')  # quotechar='', quoting=csv.QUOTE_MINIMAL quotechar must be set if quoting enabled
    randwriter.writerow(['fname'] + ['label'])

    files = glob.glob(
      FLAGS.data_dir + '*.wav')  # must add *, or it just regard the test folder as a file  #
    for file in files:  # in python it is very quickly, because it doesn't need UI
      filename=file.replace(FLAGS.data_dir, '') # if after run, it cause: ValueError: I/O operation on closed file
      #print('filemane:',filename)
      result = sess.run(logits, feed_dict={
          wav_data_placeholder: file})  # when test the accuracy, using logits, not softmax in train.py, to conduct tf.argmax, and then get the confusion matrix
      predicted_indices = np.argmax(result)
      randwriter.writerow([filename] + [lableList[predicted_indices]])
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs', )
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs', )
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.', )
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is', )
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices', )
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint', )


  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)', )

  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')  
  
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='/home/tonzheng/vtrain/speech_commands_train_add_sub_data/conv.ckpt-18000',
      #default='/home/tonzheng/ztSpeechtrainResult/speech_commands_train/conv.ckpt-18000',
      #'/home/tonzheng/S/conv.ckpt-18000' default='/home/tonzheng/ztSpeechtrainResult/speech_commands_train/conv.ckpt-18000',#default='/home/tonzheng/ztSpeechtrainResult/speech_commands_train', # reuse the check point
      help='If specified, restore this pretrained model before any training.')
  
  parser.add_argument(
      '--result_dir',
      type=str,
      default='/home/tonzheng/wresult/result',
      help='Directory to save the result.')

  parser.add_argument(
      '--output_file',
      type=str,
      default='/home/tonzheng/ytest/result/graph_file',
      help='Directory to save the result.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/data/tonzheng/speech_test/test/audio/',#,'/home/tonzheng/ytest/myValidaion/'
      help="""\
      Where to download the speech training data to.
      """)


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

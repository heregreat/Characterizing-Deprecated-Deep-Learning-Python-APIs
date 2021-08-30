# 1. Instead creat method for every model, they creat one base Class like keras.layers.RNN, then for every other models samilar to RNN, creat class to include keras.layers.RNN.
bidirectional_dynamic_rnn
dynamic_rnn
static_state_saving_rnn
static_bidirectional_rnn
"Please use `keras.layers.RNN(cell, unroll=True)`, "which is equivalent to this API"

"Please use `keras.layers.Bidirectional(keras.layers.RNN(cell))`, which is equivalent to this API")

# 2. Change method to class without changing usage
## layers.conv2d
- layers.conv3d
- separable_conv1d
- separable_conv2d
- conv2d_transpose
- conv3d_transpose

/tensorflow-r2.0/tensorflow/python/layers/convolutional.py

Deprecated: Use `tf.keras.layers.Conv2D` instead.

# 3.Tensorflow has several different versions for some methods. 

## Sometimes they define different name for these versions, like, `tf.compat.v1` -> `tf.compat.v2`, sometimes they delete these version name, like `tf.compat.v1.random.stateless_multinomial` -> `tf.random.stateless_categorical`, sometimes they add version name,like `tf.graph_util.must_run_on_cpu` -> `tf.compat.v1.graph_util.must_run_on_cpu`.

## 22. tf.compat.v1.nn.softmax_cross_entropy_with_logits

fractional_max_pool
Future major versions of TensorFlow will allow gradients to flow into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.

## 7. tf.compat.v1.random.stateless_multinomial
tensorflow-r2.0/tensorflow/python/ops/stateless_random_ops.py

Use tf.random.stateless_categorical instead.
## 6. tf.graph_util.must_run_on_cpu
- tf.extract_sub_graph

/tensorflow-r2.0/tensorflow/python/framework/graph_util_impl.py

Deprecated: Use `tf.compat.v1.graph_util.must_run_on_cpu`

# 4.encapsulation Made single method that does all casting
## 4.to_float
- to_double
- to_int32
- to_int64
- to_bfloat16
- to_complex64
- to_complex128

use tf.cast instead.

    tf.cast can achieve the function of all the APIs above
# 5.encapsulation creat new module that contains several methods(may not specific to Tensorflow)
## 24. histogram_summary
- image_summary
- audio_summary
- merge_summary
- merge_all_summaries
- scalar_summary

Please switch to tf.summary.histogram. Note that "
    "tf.summary.histogram uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in.

# 6.Contrib Module
In TensorFlow 1.X, contrib module store API from third party project which is maintained by other community members. This may partly because TensorFlow relied on community in the eary stage.
Contrib Module is deleted in TensorFlow 2.0

# **Default args' value name change**

## 1. tf.nn.conv1d(input, filters, stride, padding,data_format='NHWC', dilations=None, name=None)

tensorflow-r2.0/tensorflow/python/ops/nn_ops.py

Deprecated: "`NHWC` for data_format is deprecated, use `NWC` instead"

# **Migrating APIs to another modules because of the deleting of some packages like contrib**
## 1. tf.contrib.looses.softmax_cross_entropy
- compute_weighted_loss
- add_loss
- get_losses
- get_regularization_losses
- get_total_loss
- absolute_difference
- sigmoid_cross_entropy
- softmax_cross_entropy
- sparse_softmax_cross_entropy
- log_loss
- hinge_loss
- mean_squared_error
- mean_pairwise_squared_error
- tensorflow/contrib/losses/python/losses/loss_ops.py

 "Use `tf.losses.softmax_cross_entropy` instead. Note that the order of the logits and labels arguments has been changed.")

## **.........etc**

## 2. kl_divergence
- cross_entropy

The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.distributions`

# API Optimization

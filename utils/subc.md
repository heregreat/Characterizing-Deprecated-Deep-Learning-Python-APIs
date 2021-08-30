# 1. Instead creat method for every model, they creat one base Class like keras.layers.RNN, then for every other models samilar to RNN, creat class to include keras.layers.RNN.
bidirectional_dynamic_rnn
dynamic_rnn
static_state_saving_rnn
static_bidirectional_rnn
"Please use `keras.layers.RNN(cell, unroll=True)`, "which is equivalent to this API"

"Please use `keras.layers.Bidirectional(keras.layers.RNN(cell))`, which is equivalent to this API")


# 2.Tensorflow has several different versions for some methods. 

## Sometimes they define different name for these versions, like, `tf.compat.v1` -> `tf.compat.v2`, sometimes they delete these version name, like `tf.compat.v1.random.stateless_multinomial` -> `tf.random.stateless_categorical`, sometimes they add version name,like `tf.graph_util.must_run_on_cpu` -> `tf.compat.v1.graph_util.must_run_on_cpu`.

## 15. py_func

tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.

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

## 22. compute_gradient
- compute_gradient_error
- fractional_avg_pool

Use tf.test.compute_gradient in 2.0, which has better "
    "support for functions. Note that the two versions have different usage, "
    "so code change is needed.

# 3.encapsulation Made single method that does all casting
## 4.to_float
- to_double
- to_int32
- to_int64
- to_bfloat16
- to_complex64
- to_complex128

use tf.cast instead.

    tf.cast can achieve the function of all the APIs above

## 7. cpu
- gpu

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/ops.py

    Use tf.identity instead.


## 8. scalar
- vector
- matrix

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/tensor_shape.py
Use tf.TensorShape([])

# 4.encapsulation creat new module that contains several methods(may not specific to Tensorflow)


## 2. export_saved_model

- **load_from_saved_model**()

/tensorflow-r2.0/tensorflow/python/keras/saving/saved_model_experimental.py

Deprecated: Please use `model.save()` or `tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)`.

    1. as_text: bool, `False` by default. Whether to write the `SavedModel` proto
      in text format. Currently unavailable in serving-only mode.

    2. save_format: Either 'tf' or 'h5', indicating whether to save the model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.
## 13. listdiff
- setdiff1d

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/array_ops.py

This op will be removed after the deprecation date. "
                        "Please switch to tf.setdiff1d()

This op will be removed after the deprecation date. "
                        "Please switch to tf.sets.difference()   

## Category: Encapsulation Creat New Module that Contains Several Methods
- histogram_summary
- image_summary
- audio_summary
- merge_summary
- merge_all_summaries
- scalar_summary

## Deprecation Message: 
Please switch to tf.summary.histogram. Note that "
    "tf.summary.histogram uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in.

# 5. better read write semantics

## 3. alias_inplace_update
- alias_inplace_add
- alias_inplace_sub
- inplace_update
- inplace_add
- inplace_sub

tensorflow-r2.0/tensorflow/python/ops/inplace_ops.py

Prefer `tf.tensor_scatter_nd_add`, which offers the same functionality with well-defined read-write semantics.

# 6. Function replaced by other method

## 1. multi_gpu_model
tensorflow-r2.0/tensorflow/python/keras/utils/multi_gpu_utils.py

Deprecated: '2020-04-01', 'Use `tf.distribute.MirroredStrategy` instead.'

    tf.distribute.MirroredStrategy(devices=None,
    cross_device_ops=None)

## 9. tf.compat.v1.train.start_queue_runners
- add_queue_runner

THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: To construct input pipelines, use the tf.data module.

## 10. limit_epochs
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/training/input.py

Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)

## 12. tf_record_iterator
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/lib/io/tf_record.py

Use eager execution and: \n"
                  "`tf.data.TFRecordDataset(path)
## 18. count_up_to
Prefer Dataset.range instead.

## 19. **tf.compat.v1.sparse_to_dense**??

THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Create a tf.sparse.SparseTensor and use tf.sparse.to_dense instead.

## 20. init
- export

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/contrib/session_bundle/exporter.py

"No longer supported. Switch to SavedModel immediately.


## 23. create_partitioned_variables
- multinomial

Use `tf.get_variable` with a partitioner set.

Use `tf.random.categorical` instead.
## 17.initialized_value
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/variables.py

Use Variable.read_value. Variables in 2.X are initialized "
      "automatically both in eager and graph (inside tf.defun) contexts.
# 7. split the function of one method and change to other form

## 14. batch_gather
- quantize_v2

tf.batch_gather` is deprecated, please use `tf.gather` "
    "with `batch_dims=-1` instead.

## 16. clip_by_average_norm
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/clip_ops.py

"clip_by_average_norm is deprecated in TensorFlow 2.0. Please "
    "use clip_by_norm(t, clip_norm * tf.cast(tf.size(t), tf.float32), name) "
    "instead."

## 25. map_and_batch

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/data/experimental/ops/batching.py

Use `tf.data.Dataset.map(map_func, num_parallel_calls)` followed by "
    "`tf.data.Dataset.batch(batch_size, drop_remainder)`. Static tf.data "
    "optimizations will take care of using the fused implementation."
# **API Improvement**

### **Finding: The new method is more easily to use and its scope is larger.**
### **Finding: The Model save format changed, the export_saved_model cannot meet the requirement. Using model.save() is intuitive and easy for developers.**


## 6. get_backward_walk_ops

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/contrib/graph_editor/select.py


## 11. update_checkpoint_state
- checkpoint_exists
- get_checkpoint_mtimes
- remove_checkpoint

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/training/checkpoint_management.py


## 26. toco_convert
- from_session
- from_frozen_graph
- from_saved_model
- from_keras_model_file
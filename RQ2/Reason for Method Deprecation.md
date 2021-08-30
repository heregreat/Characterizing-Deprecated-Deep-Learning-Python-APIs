# Tensorflow 2.3

Analyzed 235 deprecated APIs, 199 are method deprecation, 36 are parameter deprecation.

Agreement with Max's result: 209/235 (88.9%)agreement

# **API Optimization (107, 53.77%)**

### **Finding: The new method is more easy to use and its scope is larger.**
### **Finding: The Model save format changed, the export_saved_model cannot meet the requirement. Using model.save() is intuitive and easy for developers.**

## 1. multi_gpu_model
tensorflow-r2.0/tensorflow/python/keras/utils/multi_gpu_utils.py

Deprecated: '2020-04-01', 'Use `tf.distribute.MirroredStrategy` instead.'

    tf.distribute.MirroredStrategy(devices=None,
    cross_device_ops=None)


## 2. export_saved_model

- **load_from_saved_model**()

/tensorflow-r2.0/tensorflow/python/keras/saving/saved_model_experimental.py

Deprecated: Please use `model.save()` or `tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None)`.

    1. as_text: bool, `False` by default. Whether to write the `SavedModel` proto
      in text format. Currently unavailable in serving-only mode.

    2. save_format: Either 'tf' or 'h5', indicating whether to save the model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.

## 4. alias_inplace_update
- alias_inplace_add
- alias_inplace_sub
- inplace_update
- inplace_add
- inplace_sub

tensorflow-r2.0/tensorflow/python/ops/inplace_ops.py

Prefer `tf.tensor_scatter_nd_add`, which offers the same functionality with well-defined read-write semantics.

## 10.to_float
- to_double
- to_int32
- to_int64
- to_bfloat16
- to_complex64
- to_complex128

use tf.cast instead.

    tf.cast can achieve the function of all the APIs above

## 17. tf.nn.static_rnn
- bidirectional_dynamic_rnn
- dynamic_rnn
- static_state_saving_rnn
- static_bidirectional_rnn

"Please use `keras.layers.RNN(cell, unroll=True)`, "which is equivalent to this API"

"Please use `keras.layers.Bidirectional(keras.layers.RNN(cell))`, which is equivalent to this API")

## 22. get_backward_walk_ops

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/contrib/graph_editor/select.py

## 23. cpu
- gpu

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/ops.py

    Use tf.identity instead.

## 25. scalar
- vector
- matrix

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/tensor_shape.py
Use tf.TensorShape([])

## 28. tf.compat.v1.train.start_queue_runners
- add_queue_runner

THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: To construct input pipelines, use the tf.data module.

## 30. limit_epochs
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/training/input.py

Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)

## 31. update_checkpoint_state
- checkpoint_exists
- get_checkpoint_mtimes
- remove_checkpoint

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/training/checkpoint_management.py

## 35. tf_record_iterator
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/lib/io/tf_record.py

Use eager execution and: \n"
                  "`tf.data.TFRecordDataset(path)

## 36. listdiff
- setdiff1d

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/array_ops.py

This op will be removed after the deprecation date. "
                        "Please switch to tf.setdiff1d()

This op will be removed after the deprecation date. "
                        "Please switch to tf.sets.difference()                       

## 38. batch_gather
- quantize_v2

tf.batch_gather` is deprecated, please use `tf.gather` "
    "with `batch_dims=-1` instead.

## 40. py_func

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

## 41. clip_by_average_norm
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/clip_ops.py

"clip_by_average_norm is deprecated in TensorFlow 2.0. Please "
    "use clip_by_norm(t, clip_norm * tf.cast(tf.size(t), tf.float32), name) "
    "instead."

## 42.initialized_value
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/variables.py

Use Variable.read_value. Variables in 2.X are initialized "
      "automatically both in eager and graph (inside tf.defun) contexts.

## 43. count_up_to
Prefer Dataset.range instead.

## 44. **tf.compat.v1.sparse_to_dense**??

THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Create a tf.sparse.SparseTensor and use tf.sparse.to_dense instead.

## 45. init
- export

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/contrib/session_bundle/exporter.py

"No longer supported. Switch to SavedModel immediately.

## 46. sample_distorted_bounding_box

`seed2` arg is deprecated.'
    'Use sample_distorted_bounding_box_v2 instead.

# 47. compute_gradient
- compute_gradient_error
- fractional_avg_pool

Use tf.test.compute_gradient in 2.0, which has better "
    "support for functions. Note that the two versions have different usage, "
    "so code change is needed.

## 50. tf.compat.v1.nn.softmax_cross_entropy_with_logits
- fractional_max_pool

Future major versions of TensorFlow will allow gradients to flow into the labels input on backprop by default.

See `tf.nn.softmax_cross_entropy_with_logits_v2`.

`seed2` and `deterministic` "
                        "args are deprecated.  Use fractional_max_pool_v2.

## 52. create_partitioned_variables
- multinomial

Use `tf.get_variable` with a partitioner set.

Use `tf.random.categorical` instead.

## 54. histogram_summary
- image_summary
- audio_summary
- merge_summary
- merge_all_summaries
- scalar_summary

Please switch to tf.summary.histogram. Note that "
    "tf.summary.histogram uses the node name instead of the tag. "
    "This means that TensorFlow will automatically de-duplicate summary "
    "names based on the scope they are created in.

## 60. map_and_batch

/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/data/experimental/ops/batching.py

Use `tf.data.Dataset.map(map_func, num_parallel_calls)` followed by "
    "`tf.data.Dataset.batch(batch_size, drop_remainder)`. Static tf.data "
    "optimizations will take care of using the fused implementation."

## 61-65. toco_convert
- from_session
- from_frozen_graph
- from_saved_model
- from_keras_model_file

## 66-107
- VARIABLES
- graph
- quantize
- kl_divergence
- crop_and_resize_v1
  
- fit_generator
- evaluate_generator
- predict_generator
- is_gpu_available
- auc
- read_data_sets

- predict_proba
- predict_classes
- map_fn_v2
- initialize

- random_binomial
- get_updates_for
- get_losses_for
- run_multiple_tasks_in_processes
- stream_stderr
- experimental_make_numpy_dataset
- legacy_snapshot
- get_next_as_optional
- start_tracing
- monitor
- start_profiler_server
- join_independent_workers

- argmin
- argmax,
- increment_var
- shard


# API Name Change (58, 29.15%)
# **Cleaning API alias or change API name to make it more intuitive and easy to use. **

### **Finding: Some methods in TensorFlow have many alias, clearing some alias will make the method more organized and concise.**
### **Finding: Method name change, assign is more intuitive and proper than load.**

## 1. add_variable(self, *args, **kwargs)

- apply(self, inputs, *args, **kwargs)

tensorflow-r2.0/tensorflow/python/keras/engine/base_layer.py

Deprecated: Please use `layer.add_weight` method instead.


    Alias for `add_weight`
        return self.add_weight(*args, **kwargs)


## 3. tf.Variable.load(value, session=None)

Deprecated: Prefer `Variable.assign` which has equivalent behavior in 2.X.

## 4. layers.conv2d
- layers.conv3d
- separable_conv1d
- separable_conv2d
- conv2d_transpose
- conv3d_transpose

/tensorflow-r2.0/tensorflow/python/layers/convolutional.py

Deprecated: Use `tf.keras.layers.Conv2D` instead.

## 10. layer.max_pooling2d

- max_pooling3d
- max_pooling1d
- average_pooling3d
- average_pooling2d
- average_pooling1d

/tensorflow-r2.0/tensorflow/python/layers/pooling.py

Deprecated: Use `keras.layers.MaxPooling2D` instead.

## 16. layer.dense

- layer.flatten

/tensorflow-r2.0/tensorflow/python/layers/core.py

Deprecated: Use `keras.layers.Dense` instead.

## 18. tf.graph_util.must_run_on_cpu
- tf.extract_sub_graph

/tensorflow-r2.0/tensorflow/python/framework/graph_util_impl.py

Deprecated: Use `tf.compat.v1.graph_util.must_run_on_cpu`

## 20. tf.compat.v1.random.stateless_multinomial
tensorflow-r2.0/tensorflow/python/ops/stateless_random_ops.py

Use tf.random.stateless_categorical instead.

## 21. batch_normalization
tensorflow-r2.0/tensorflow/python/layers/normalization.py

## 22. tf.compat.v1.load_file_system_library
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/load_library.py

    Use `tf.load_library` instead.
## 23. test_session
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/test_util.py
Use `self.session()` or `self.cached_session()` instead.

## 24. VARIABLES
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/ops.py

Use `tf.GraphKeys.GLOBAL_VARIABLES` instead

## 25. tensor_shape_from_node_def_name
- convert_variables_to_constants
- remove_training_nodes

## 28. match_filenames_once
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/training/input.py
train.match_filenames_once

## 29. substr_deprecated
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/string_ops.py

Use `tf.strings.substr` instead of `tf.substr`

## 30. create_summary_file_writer

Renamed to create_file_writer()

## 31. all_variables
- initialize_variables
- initialize_all_variables
- initialize_local_variables

Use `tf.local_variables_initializer` instead.

## 35. initialize_all_tables

Use `tf.tables_initializer` instead.

## 36. batch_scatter_update
Use the batch_scatter_update method of Variable instead.

## 37. var_scope
The .var_scope property is deprecated. Please change your "
              "code to use the .variable_scope property

## 38. div
Deprecated in favor of operator or tf.math.divide

## 39-58. sparse_average_precision_at_k
- sparse_precision_at_k
Use average_precision_at_k instead

- Print
- map_and_batch_with_legacy_function
- experimental_ref
- experimental_run_v2
- while_loop_v2
- foldl_v2
- foldr_v2
- scan_v2

- experimental_run_functions_eagerly
- experimental_functions_run_eagerly
- where

# Compatibility Issiue (19, 9.55%)
## **Some methods have compatibility issiue in newer version, so they will be removed **

## 1. tf.compat.v1.saved_model.build_tensor_info(tensor)
Following APIs have the same problem:
- get_tensor_from_tensor_info()
- main_op()
- main_op_with_restore()
- simple_save()

Deprecated: This function will only be available through the v1 compatibility library as `tf.compat.v1.saved_model.utils.build_tensor_info` or `tf.compat.v1.saved_model.build_tensor_info`.

## 6. tf.compat.v1.train.input_producer（Also API Improvement）
Following APIs have the same problem:
- string_input_producer
- range_input_producer
- slice_input_producer
- batch
- maybe_batch
- batch_join
- maybe_batch_join
- shuffle_batch
- maybe_shuffle_batch
- shuffle_batch_join
- maybe_shuffle_batch_join

tensorflow-r2.0/tensorflow/python/training/input.py

**Deprecated: Queue-based input pipelines have been replaced by tf.data. Use tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs). If shuffle=False, omit the .shuffle(...).**

    Input pipelines based on Queues are not supported when eager execution is enabled. Please use the tf.data API to ingest data under eager execution.

## 18-19. disable_resource_variables
tensorflow-r2.0/tensorflow/python/ops/variable_scope.py

non-resource variables are not supported in the long term

    If your code needs tf.disable_resource_variables() to be called to work properly please file a bug.

- make_saveable_from_iterator


# Feature Deleting (15, 7.54%)
## **No longer needed due to feature deleting **

## 1. tf.compat.v1.layers.Layer.graph

tensorflow-r2.0/tensorflow/python/layers/base.py

    Stop using this property because tf.layers layers no longer track their graph.

    # We no longer track graph in tf.layers layers. This property is only kept to maintain API backward compatibility.
## 2. sparse_merge
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/ops/sparse_ops.py

    No similar op available at this time.

# 3-15
- do_quantize_training_on_graphdef
- graph_parents
- convert_all_kernels_in_model

- set_learning_phase
- learning_phase_scope
- terminate_keras_multiprocessing_pools
- updates
- state_updates
- start
- stop
- maybe_create_event_file
- save
- colocate_with




- to_float_32

- to_int_32

- max_pooling_2d

- dense
tf.layers.dense` is deprecated and '
                'will be removed in a future version. '
                'Please use `tf.keras.layers.Dense` instead.

- soft_max_cross_entropy_logits

- tf.compat.v1.sparse_to_dense

    Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Create a tf.sparse.SparseTensor and use tf.sparse.to_dense instead.

- tf.layers.batch_normalization
warnings.warn(
      '`tf.layers.batch_normalization` is deprecated and '
      'will be removed in a future version. '
      'Please use `tf.keras.layers.BatchNormalization` instead. '
      'In particular, `tf.control_dependencies(tf.GraphKeys.UPDATE_OPS)` '
      'should not be used (consult the `tf.keras.layers.BatchNormalization` '
      'documentation).')

- tf.data.experimental.map_and_batch

    Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use tf.data.Dataset.map(map_func, num_parallel_calls) followed by tf.data.Dataset.batch(batch_size, drop_remainder). Static tf.data optimizations will take care of using the fused implementation.

- test_session

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use self.session() or self.cached_session() instead.

- py_func
    @deprecation.deprecated(
    date=None,
    instructions="""tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    """)

- parallel_interleave

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE) instead. If sloppy execution is desired, use tf.data.Options.experimental_deterministic.


-load
false positive
not tf.saved_model.load

- where

- shard
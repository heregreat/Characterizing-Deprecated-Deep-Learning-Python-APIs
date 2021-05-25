 - tf_record_iterator:
 tf.compat.v1.io.tf_record_iterator

 An iterator that read the records from a TFRecords file. (deprecated)

 Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use eager execution and: tf.data.TFRecordDataset(path)

 - make_one_shot_iterator
 Creates an iterator for elements of this dataset. (deprecated)

 Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: This is a deprecated API that should only be used in TF 1 graph mode and legacy TF 2 graph mode available through tf.compat.v1. In all other situations -- namely, eager mode and inside tf.function -- you can consume dataset elements using for elem in dataset: ... or by explicitly creating iterator via iterator = iter(dataset) and fetching its elements via values = next(iterator). Furthermore, this API is not available in TF 2. During the transition from TF 1 to TF 2 you can use tf.compat.v1.data.make_one_shot_iterator(dataset) to create a TF 1 graph mode style iterator for a dataset created through TF 2 APIs. Note that this should be a transient state of your code base as there are in general no guarantees about the interoperability of TF 1 and TF 2 code.

 - is_gpu_available
 Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version. Instructions for updating: Use tf.config.list_physical_devices('GPU') instead.

 
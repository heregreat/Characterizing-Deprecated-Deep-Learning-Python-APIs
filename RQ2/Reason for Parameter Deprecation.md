# Unnecessary Parameter (3, 8.33%)
## **Some parametas can be computed automatically and aren't needed.**
## 1. add_update(self, updates, input)
tensorflow-r2.0/tensorflow/python/keras/engine/base_layer.py

Args deprecated: become deprecated in TensorFlow 1.15, haven't removed in latest version.

      inputs: Deprecated, will be automatically inferred.

      if call_context.in_call:
      relevant_inputs = call_context.inputs
    else:
      inbound_nodes = getattr(self, '_inbound_nodes', [])
      relevant_inputs = [node.input_tensors for node in inbound_nodes]

## 2. import_graph_def
    Please file an issue at https://github.com/tensorflow/tensorflow/issues if you depend on this feature.', 'op_dict'

## 3. create_op
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/python/framework/ops.py

        create_op(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True)

Shapes are always computed; don't use the compute_shapes as it has no effect.", "compute_shapes"

# Parameter Name Change (33, 91.67%)
## **Change Variable to Another Euaivalent Parameter**

## 1. **add_meta_graph**(self, tags, signature_def_map=None, assets_collection=None, legacy_init_op=None, clear_devices=False, main_op=None, strip_default_attrs=False, saver=None)
- **add_meta_graph_and_variables**():

tensorflow/python/saved_model/builder_impl.py

Args deprecated: "Pass your op to the equivalent parameter main_op instead.", "legacy_init_op"

## 3. tf.nn.dropout()
tensorflow-r2.0/tensorflow/python/ops/nn_ops.py

Deprecated: SOME ARGUMENTS ARE DEPRECATED: (keep_prob). They will be removed in a future version. Instructions for updating: Please use rate instead of keep_prob. Rate should be set to rate = 1 - keep_prob.

        tf.nn.dropout(
            x,
            keep_prob=None,
            noise_shape=None,
            seed=None,
            name=None,
            rate=None
        )

## 4. tf.expand_dims
tensorflow-r2.0/tensorflow/python/ops/array_ops.py

Deprecated: Use the `axis` argument instead", "dim"

        tf.expand_dims(input, axis=None, name=None, dim=None)

## 5. sparse_split
tensorflow-r2.0/tensorflow/python/ops/sparse_ops.py

split_dim is deprecated, use axis instead", "split_dim"

        sparse_split(keyword_required=KeywordRequired(),
                 sp_input=None,
                 num_split=None,
                 axis=None,
                 name=None,
                 split_dim=None):

## 6.sparse_concat
tensorflow-r2.0/tensorflow/python/ops/sparse_ops.py

"concat_dim is deprecated, use axis instead", "concat_dim")

        def sparse_concat(axis,
                  sp_inputs,
                  name=None,
                  expand_nonconcat_dim=False,
                  concat_dim=None,
                  expand_nonconcat_dims=None):
## 7. decode_raw_v1
tensorflow-r2.0/tensorflow/python/ops/parsing_ops.py

"bytes is deprecated, use input_bytes instead","bytes")

    def decode_raw_v1(
        input_bytes=None,
        out_type=None,
        little_endian=True,
        name=None,
        bytes=None  # pylint: disable=redefined-builtin)

## 8. argmax
- argmin

/tensorflow-r2.0/tensorflow/python/ops/math_ops.py

"Use the `axis` argument instead","dimension"

    def argmax(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64)
## 10. reduce_sum_v1
"keep_dims is deprecated, use keepdims instead", "keep_dims"

## 11. seek
"position is deprecated in favor of the offset argument.",
      "position"

      seek(self, offset=None, whence=0, position=None)
## 12. cosine_distance
/Users/nianliu/数据/tensorflow-r2.0/tensorflow/contrib/losses/python/losses/loss_ops.py

"dim is deprecated, use axis instead", "dim"

## 13. squeeze
- reverse_sequence
- extract_image_patches
Use the `axis` argument instead",
                             "squeeze_dims

## 14. string_split
"delimiter is deprecated, please use sep instead.",
                             "delimiter"
## 15. weighted_cross_entropy_with_logits
- l2_normalize
targets is deprecated, use labels instead", "targets"

## 17.sparse_concat
- sparse_reduce_max
- sparse_reduce_max_sparse
- sparse_reduce_sum
- sparse_reduce_sum_sparse

These four APIs also have endpoint deprecation.

@deprecation.deprecated_endpoints("sparse_reduce_sum_sparse")

@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")

concat_dim is deprecated, use axis instead", "concat_dim"

## 22. crop_and_resize_v1
- cond
- log_softmax
- softmax_cross_entropy_with_logits_v2_helper

'box_ind is deprecated, use box_indices instead',
                             'box_ind'

## 26-33. norm
- count_nonzero
- reduce_prod_v1
- reduce_min_v1
- reduce_max_v1
- reduce_all_v1
- reduce_any_v1
- reduce_logsumexp_v1

'keep_dims is deprecated, use keepdims instead', 'keep_dims'
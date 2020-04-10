import tensorflow as tf


def CER(y_true, y_pred):
    """
    Average edit distance, ignoring padding of0s.
    Score returned is the Levenshtein edit distance divided by the total length
    of reference truth.

    Parameters
    ----------
    y_pred : Tensor[shape=(batch_size, prediction_seq_length, num_classes)]
        softmax output from model
    y_true : Tensor[shape=(batch_size, labels_seq_length, num_classes)]
        one-hot vector of ground truth y_true

    Returns
    -------
    cer : float,
        character error rate
    """
    # reference_length = y_true.shape[1]
    reference_length = tf.cast(tf.shape(y_true)[1], tf.float32)

    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    nonzero_idx = tf.where(tf.not_equal(y_pred, 0))
    sparse_outputs = tf.SparseTensor(nonzero_idx,
                                     tf.gather_nd(y_pred, nonzero_idx),
                                     tf.shape(y_pred, out_type=tf.int64))

    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
    nonzero_idx = tf.where(tf.not_equal(y_true, 0))
    label_sparse_outputs = tf.SparseTensor(nonzero_idx,
                                           tf.gather_nd(y_true, nonzero_idx),
                                           tf.shape(y_true, out_type=tf.int64))

    distance = tf.reduce_sum(
        tf.edit_distance(sparse_outputs, label_sparse_outputs, normalize=False)
    )

    cer = tf.math.divide(distance, reference_length)
    return cer

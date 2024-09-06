import tensorflow as tf

from layers import down, up, fuse


def DeepCrack(input_shape: tuple[int, int, int, int]) -> tf.keras.Model:
    """
    A functional implementation of the DeepCrack model.

    Note that the network in the paper differs slightly from the code by using batch normalization like SegNet and the paper indicate.
    """
    input = tf.keras.layers.Input(shape=input_shape[1:], batch_size=input_shape[0])

    # Encoder
    x1, residue_1, argmax_down_1, unpooled_shape_1 = down(input, num_convolutions=2, filters=64)
    x2, residue_2, argmax_down_2, unpooled_shape_2 = down(x1, num_convolutions=2, filters=128)
    x3, residue_3, argmax_down_3, unpooled_shape_3 = down(x2, num_convolutions=3, filters=256)
    x4, residue_4, argmax_down_4, unpooled_shape_4 = down(x3, num_convolutions=3, filters=512)
    x5, residue_5, argmax_down_5, unpooled_shape_5 = down(x4, num_convolutions=3, filters=512)

    # Decoder
    x6 = up(x5, num_convolutions=3, filters=512, max_indices=argmax_down_5, unpooled_shape=unpooled_shape_5, reduce_last=False)
    x7 = up(x6, num_convolutions=3, filters=512, max_indices=argmax_down_4, unpooled_shape=unpooled_shape_4, reduce_last=True)
    x8 = up(x7, num_convolutions=3, filters=256, max_indices=argmax_down_3, unpooled_shape=unpooled_shape_3, reduce_last=True)
    x9 = up(x8, num_convolutions=2, filters=128, max_indices=argmax_down_2, unpooled_shape=unpooled_shape_2, reduce_last=True)
    x10 = up(x9, num_convolutions=2, filters=64, max_indices=argmax_down_1, unpooled_shape=unpooled_shape_1, reduce_last=True)

    # Skip connections
    skip1 = fuse(residue_1, x10, 1, input_shape[1:3])
    skip2 = fuse(residue_2, x9, 2, input_shape[1:3])
    skip3 = fuse(residue_3, x8, 3, input_shape[1:3])
    skip4 = fuse(residue_4, x7, 4, input_shape[1:3])
    skip5 = fuse(residue_5, x6, 5, input_shape[1:3])

    # Combine for final
    merged = tf.concat([skip5, skip4, skip3, skip2, skip1], axis=-1)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='SAME', activation='sigmoid', kernel_initializer='he_normal')(merged)
    return tf.keras.Model(inputs=[input], outputs=[outputs])

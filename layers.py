import tensorflow as tf
from tensorflow.keras import layers

def maxpooling_with_argmax(input_tensor: tf.Tensor, pool_size: tuple[int, int], strides: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    """Maxpooling with argmax layer from https://github.com/danielenricocahall/Keras-SegNet/blob/master/custom_layers/layers.py"""
    with tf.name_scope('maxpooling_with_argmax2D'):
        output, argmax = tf.nn.max_pool_with_argmax(
            input_tensor,
            ksize=[1, *pool_size, 1],
            strides=[1, *strides, 1],
            padding='SAME'
        )

        return output, argmax

def unpool(input_tensor: tf.Tensor, max_indices: tf.Tensor, unpooled_shape: tuple[int, int, int, int]) -> tf.Tensor:
    """Unpooling layer from https://github.com/danielenricocahall/Keras-SegNet/blob/master/custom_layers/layers.py"""
    with tf.name_scope('unpool2D'):
        ret = tf.scatter_nd(
            tf.expand_dims(layers.Flatten()(max_indices), -1),
            layers.Flatten()(input_tensor),
            [tf.math.reduce_prod(unpooled_shape, 0)]
        )

        return tf.reshape(ret, unpooled_shape)

def conv_relu(input_tensor: tf.Tensor, filters: int) -> tf.Tensor:
    """3x3 convolution with padding, batch normalization and ReLu activation"""
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='SAME', kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def up(input_tensor: tf.Tensor, num_convolutions: int, filters: int, max_indices: tf.Tensor, unpooled_shape: tuple[int, int, int, int], reduce_last: bool = False) -> tf.Tensor:
    """A decoder layer, represented by a sets of convolution layers and a maxunpool."""
    x = unpool(input_tensor, max_indices, unpooled_shape)

    for idx in range(num_convolutions):
        x = conv_relu(x, filters if not reduce_last or idx < num_convolutions - 1 else filters // 2)
    return x

def down(input_tensor: tf.Tensor, num_convolutions: int, filters: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple[int, int, int, int]]:
    """An encoder layer, represented by a set of convolution layers and a maxpool with argmax."""
    x = input_tensor
    for _ in range(num_convolutions):
        x = conv_relu(x, filters)

    unpooled_shape = x.shape
    pooled, argmax = maxpooling_with_argmax(x, pool_size=(2, 2), strides=(2, 2))
    return pooled, x, argmax, unpooled_shape

def fuse(down_input: tf.Tensor, up_input: tf.Tensor, scale: int, crop_shape: tuple[int, int]) -> tf.Tensor:
    merged = tf.concat([down_input, up_input], axis=3)
    x = layers.UpSampling2D(size=(2**(scale - 1), 2**(scale - 1)))(merged)
    x = conv_relu(x, 64)
    x = layers.Conv2D(1, kernel_size=(3, 3), padding='SAME', kernel_initializer='he_normal')(x)

    return x
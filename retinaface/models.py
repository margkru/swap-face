import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU


def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)



class ConvUnit(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, f, k, s, wd, act=None, **kwargs):
        super(ConvUnit, self).__init__(**kwargs)
        self.conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False)
        self.bn = BatchNormalization()

        if act is None:
            self.act_fn = tf.identity
        elif act == 'relu':
            self.act_fn = ReLU()
        elif act == 'lrelu':
            self.act_fn = LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))

    def call(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network"""
    def __init__(self, out_ch, wd, **kwargs):
        super(FPN, self).__init__(**kwargs)
        act = 'relu'
        self.out_ch = out_ch
        self.wd = wd
        if (out_ch <= 64):
            act = 'lrelu'

        self.output1 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output2 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.output3 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act)
        self.merge1 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)
        self.merge2 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act)

    def call(self, x):
        output1 = self.output1(x[0])  # [80, 80, out_ch]
        output2 = self.output2(x[1])  # [40, 40, out_ch]
        output3 = self.output3(x[2])  # [20, 20, out_ch]

        up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
        up3 = tf.image.resize(output3, [up_h, up_w], method='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
        up2 = tf.image.resize(output2, [up_h, up_w], method='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3
    
    def get_config(self):
        config = {
            'out_ch': self.out_ch,
            'wd': self.wd,
        }
        base_config = super(FPN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SSH(tf.keras.layers.Layer):
    """Single Stage Headless Layer"""
    def __init__(self, out_ch, wd, **kwargs):
        super(SSH, self).__init__(**kwargs)
        assert out_ch % 4 == 0
        self.out_ch = out_ch
        self.wd = wd
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.conv_3x3 = ConvUnit(f=out_ch // 2, k=3, s=1, wd=wd, act=None)

        self.conv_5x5_1 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_5x5_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.conv_7x7_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act)
        self.conv_7x7_3 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None)

        self.relu = ReLU()

    def call(self, x):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = self.relu(output)

        return output
    
    def get_config(self):
        config = {
            'out_ch': self.out_ch,
            'wd': self.wd,
        }
        base_config = super(SSH, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BboxHead(tf.keras.layers.Layer):
    """Bbox Head Layer"""
    def __init__(self, num_anchor, wd, **kwargs):
        super(BboxHead, self).__init__(**kwargs)
        self.num_anchor = num_anchor
        self.wd = wd
        self.conv = Conv2D(filters=num_anchor * 4, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 4])
    
    def get_config(self):
        config = {
            'num_anchor': self.num_anchor,
            'wd': self.wd,
        }
        base_config = super(BboxHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LandmarkHead(tf.keras.layers.Layer):
    """Landmark Head Layer"""
    def __init__(self, num_anchor, wd, name='LandmarkHead', **kwargs):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.wd = wd
        self.conv = Conv2D(filters=num_anchor * 10, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 10])

    def get_config(self):
        config = {
            'num_anchor': self.num_anchor,
            'wd': self.wd,
        }
        base_config = super(LandmarkHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClassHead(tf.keras.layers.Layer):
    """Class Head Layer"""
    def __init__(self, num_anchor, wd, name='ClassHead', **kwargs):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.wd = wd
        self.conv = Conv2D(filters=num_anchor * 2, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 2])

    def get_config(self):
        config = {
            'num_anchor': self.num_anchor,
            'wd': self.wd,
        }
        base_config = super(ClassHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


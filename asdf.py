import tensorflow as tf
import os
import math

bn_axis = -1
initializer = 'glorot_normal'
# initializer = tf.keras.initializers.TruncatedNormal(
#     mean=0.0, stddev=0.05, seed=None)
# initializer = tf.keras.initializers.VarianceScaling(
#     scale=0.05, mode='fan_avg', distribution='normal', seed=None)
# initializer = tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.01, seed=None)
# initializer = tf.keras.initializers.RandomUniform(
#     minval=-0.0001, maxval=0.0001, seed=None)
gammaInit = 'ones'
# gammaInit = tf.keras.initializers.Constant(value=1.0)
# maxNorm = 0.1


def residual_unit_v3(input, num_filter, stride, dim_match, name):
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_initializer=gammaInit,
                                           name=name + '_bn1')(input)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv1_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_initializer=gammaInit,
                                           name=name + '_bn2')(x)
    x = tf.keras.layers.PReLU(name=name + '_relu1',
                              alpha_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name=name + '_conv2_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3),
                               strides=stride,
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name=name + '_conv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_initializer=gammaInit,
                                           name=name + '_bn3')(x)
    if (dim_match):
        shortcut = input
    else:
        shortcut = tf.keras.layers.Conv2D(num_filter, (1, 1),
                                          strides=stride,
                                          padding='valid',
                                          kernel_initializer=initializer,
                                          use_bias=False,
                                          kernel_regularizer=tf.keras.regularizers.l2(
                                              l=5e-4),
                                          name=name + '_conv1sc')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                                      scale=True,
                                                      momentum=0.9,
                                                      epsilon=2e-5,
                                                      renorm=True,
                                                      renorm_clipping={
                                                          'rmax': 3,
                                                          'rmin': 0.3333,
                                                          'dmax': 5},
                                                      renorm_momentum=0.9,
                                                      beta_regularizer=tf.keras.regularizers.l2(
                                                          l=5e-4),
                                                      gamma_regularizer=tf.keras.regularizers.l2(
                                                          l=5e-4),
                                                      gamma_initializer=gammaInit,
                                                      name=name + '_sc')(shortcut)
    return x + shortcut


def get_fc1(input):
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_initializer=gammaInit,
                                           name='bn1')(input)
    x = tf.keras.layers.Dropout(0.4)(x)
    resnet_shape = input.shape
    x = tf.keras.layers.Reshape(
        [resnet_shape[1] * resnet_shape[2] * resnet_shape[3]], name='reshapelayer')(x)
    x = tf.keras.layers.Dense(512,
                              name='E_DenseLayer', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4),
                              bias_regularizer=tf.keras.regularizers.l2(
                                  l=5e-4))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           scale=False,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           name='fc1')(x)
    return x


def ResNet50():

    input_shape = [112, 112, 3]
    filter_list = [64, 64, 128, 256, 512]
    units = [3, 4, 14, 3]
    num_stages = 4

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ZeroPadding2D(
        padding=(1, 1), name='conv0_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(
                                   l=5e-4),
                               name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           renorm=True,
                                           renorm_clipping={'rmax': 3,
                                                            'rmin': 0.3333,
                                                            'dmax': 5},
                                           renorm_momentum=0.9,
                                           beta_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_regularizer=tf.keras.regularizers.l2(
                                               l=5e-4),
                                           gamma_initializer=gammaInit,
                                           name='bn0')(x)
    x = tf.keras.layers.PReLU(
        name='prelu0',
        alpha_regularizer=tf.keras.regularizers.l2(
            l=5e-4))(x)

    for i in range(num_stages):
        x = residual_unit_v3(x, filter_list[i + 1], (2, 2), False,
                             name='stage%d_unit%d' % (i + 1, 1))
        for j in range(units[i] - 1):
            x = residual_unit_v3(x, filter_list[i + 1], (1, 1),
                                 True, name='stage%d_unit%d' % (i + 1, j + 2))

    x = get_fc1(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name='resnet50')
    model.trainable = True
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        # if ('conv0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('bn0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('prelu0' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage1' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage2' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage3' in model.layers[i].name):
        #     model.layers[i].trainable = False
        # if ('stage4' in model.layers[i].name):
        #     model.layers[i].trainable = False

    return model


class Arcfacelayer(tf.keras.layers.Layer):
    def __init__(self, output_dim=10572, s=64., m=0.50):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        super(Arcfacelayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],
                                             self.output_dim),
                                      initializer=initializer,
                                      regularizer=tf.keras.regularizers.l2(
                                          l=5e-4),
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)

    def call(self, embedding, labels):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m  # issue 1
        threshold = math.cos(math.pi - self.m)
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / embedding_norm
        weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
        weights = self.kernel / weights_norm
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m),
                                      tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=self.output_dim, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(
            cos_mt_temp, mask), name='arcface_loss_output')

        return output

    def compute_output_shape(self, input_shape):

        return input_shape[0], self.output_dim


class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        self.resnet = ResNet50()
        # self.arcface = Arcfacelayer()

    def call(self, x, y):
        x = self.resnet(x)
        return x


model = train_model()
print(len(model.losses))
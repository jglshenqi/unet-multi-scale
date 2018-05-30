from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, \
    Dropout, convolutional, Conv2DTranspose, Concatenate, Activation, Flatten, Dense
from keras import backend
from keras.activations import softmax
import keras
from keras.optimizers import Adam
import tensorflow as tf
import help_functions
import numpy as np
from keras.layers import Lambda
import configparser

config = configparser.RawConfigParser()
config.read('./configuration.txt',encoding='utf-8')
loss_w = [1, 2, 3, 4]
loss_w[0] = float(config.get('public', 'loss_weight_0'))
loss_w[1] = float(config.get('public', 'loss_weight_1'))
loss_w[2] = float(config.get('public', 'loss_weight_2'))
loss_w[3] = float(config.get('public', 'loss_weight_3'))


def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, data_format='channels_first', padding='same')(x)

    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', data_format='channels_first', use_bias=False,
                        activation=None)(x)

    return x


def side_branch2(x, factor):
    x = Conv2D(1, (1, 1), activation=None, data_format='channels_first', padding='same')(x)

    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(2, kernel_size, strides=factor, padding='same', data_format='channels_first', use_bias=False,
                        activation=None)(x)

    return x


def side_branch3(x, factor):
    # x = Conv2D(32, (3, 3), activation=None, data_format='channels_first', padding='same')(x)
    x = Conv2D(2, (1, 1), activation=None, data_format='channels_first', padding='same')(x)

    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(2, kernel_size, strides=factor, padding='same', data_format='channels_first', use_bias=False,
                        activation=None)(x)

    return x


def slice_0(x):
    return x[:, 0:1, :, :]


def slice_1(x):
    return x[:, 1:2, :, :]


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(backend.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                backend.batch_set_value(weight_value_tuples)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def Boundary_Refinement(inside):
    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same', data_format='channels_first')(inside)
    conv2 = Conv2D(2, (3, 3), padding='same', data_format='channels_first')(conv1)
    fuse = concatenate([conv2, inside], axis=1)
    fuse = Conv2D(2, (1, 1), data_format='channels_first')(fuse)
    return fuse


def Boundary_Refinement2(inside):
    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same', data_format='channels_first')(inside)
    conv2 = Conv2D(2, (3, 3), padding='same', data_format='channels_first')(conv1)
    add = keras.layers.Add()([inside, conv2])
    return add


def get_fcnet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))

    conv1 = Conv2D(64, (4, 4), activation='relu', padding='valid', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D((2, 2), padding='valid', data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)

    pool2_delt = Flatten()(pool2)
    conv3 = Dense(512, activation='relu')(pool2_delt)
    conv4 = Dense(512, activation='relu')(conv3)
    conv5 = Dense(2, activation='relu')(conv4)

    ############
    conv6 = core.Activation('softmax')(conv5)

    model = Model(input=inputs, output=conv6)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_unet(n_ch, patch_height, patch_width):
    print("=====Using unet original=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "pool1 = ", pool1.shape)
    #
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    print("conv3 = ", conv3.shape)

    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)

    print("conv6 = ", conv6.shape)
    ############
    conv7 = core.Activation('softmax', name='last')(conv6)
    print("conv7 = ", conv7.shape)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
            #target_tensors=None)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_unet2(n_ch, patch_height, patch_width):
    print("=====Using unet2 (sigmoid)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch(conv3, 4)
    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch(conv4, 2)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    # out1 = core.Activation('softmax',name='o1')(out1)
    # out2 = core.Activation('softmax',name='o2')(out3)
    # out3 = core.Activation('softmax',name='o3')(out2)
    # fuse = core.Activation('softmax',name='ofuse')(fuse)

    # outputs
    activ = "sigmoid"
    loss = 'categorical_crossentropy'
    out1 = Activation(activ, name='o1')(out1)
    out2 = Activation(activ, name='o2')(out2)
    out3 = Activation(activ, name='o3')(out3)
    fuse = Activation(activ, name='last')(fuse)
    #
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss={'o1': loss,
                                         'o2': loss,
                                         'o3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet3(n_ch, patch_height, patch_width):
    print("=====Using unet3 (softmax)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "pool1 = ", pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 4)
    print("conv3 = ", conv3.shape, "out1", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 2)
    print("up1 = ", up1.shape, "conv4 = ", conv4.shape, "out2 = ", out2.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2.shape, "conv5 = ", conv5.shape, "out3 = ", out3.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "fuse = ", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    # loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet5l(n_ch, patch_height, patch_width):
    loss_w = [1, 1, 1, 1, 1]
    print("=====Using new1 (5 losses)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv_a1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv_a1 = Dropout(0.2)(conv_a1)
    conv_a1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a1)
    print("conv_a1 = ", conv_a1.shape, "pool1 = ", pool1.shape)
    #
    conv_a2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv_a2 = Dropout(0.2)(conv_a2)
    conva_2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a2)
    print("conv_a2 = ", conv_a2.shape, "pool2 = ", pool2.shape)
    #
    conv_a3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv_a3 = Dropout(0.2)(conv_a3)
    conv_a3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a3)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a3)
    print("conv_a3 = ", conv_a3.shape, "pool3 = ", pool3.shape)

    conv_a4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv_a4 = Dropout(0.2)(conv_a4)
    conv_a4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a4)
    out1 = side_branch2(conv_a4, 8)
    print("conv_a4 = ", conv_a4.shape, "out1 = ", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_a4)
    up1 = concatenate([conv_a3, up1], axis=1)
    conv_b3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv_b3 = Dropout(0.2)(conv_b3)
    conv_b3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b3)
    out2 = side_branch2(conv_b3, 4)
    print("up1 = ", up1.shape, "conv_b3 = ", conv_b3.shape, "out2 = ", out2.shape)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_b3)
    up2 = concatenate([conv_a2, up2], axis=1)
    conv_b2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv_b2 = Dropout(0.2)(conv_b2)
    conv_b2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b2)
    out3 = side_branch2(conv_b2, 2)
    print("up2 = ", up2.shape, "conv_b2 = ", conv_b2.shape, "out2 = ", out3.shape)
    #
    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_b2)
    up3 = concatenate([conv_a1, up3], axis=1)
    conv_b1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv_b1 = Dropout(0.2)(conv_b1)
    conv_b1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b1)
    out4 = side_branch2(conv_b1, 1)
    print("up2 = ", up2.shape, "conv_b1 = ", conv_b1.shape, "out3 = ", out4.shape)
    #
    fuse = concatenate([out1, out2, out3, out4], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    out4 = core.Permute((2, 3, 1))(out4)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    out4 = core.Activation('softmax', name='o4')(out4)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    out4 = core.Permute((3, 1, 2), name='oo4')(out4)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "out4 = ", out4.shape, "fuse = ",
          fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, out4, fuse])
    loss = cross_entropy_balanced
    # loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'oo4': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet6l(n_ch, patch_height, patch_width):
    loss_w = [1, 1, 1, 1, 1, 1]
    print("=====Using new2 (6 losses)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv_a1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv_a1 = Dropout(0.2)(conv_a1)
    conv_a1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a1)
    print("conv_a1 = ", conv_a1.shape, "pool1 = ", pool1.shape)
    #
    conv_a2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv_a2 = Dropout(0.2)(conv_a2)
    conv_a2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a2)
    print("conv_a2 = ", conv_a2.shape, "pool2 = ", pool2.shape)
    #
    conv_a3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv_a3 = Dropout(0.2)(conv_a3)
    conv_a3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a3)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a3)
    print("conv_a3 = ", conv_a3.shape, "out1", pool3.shape)

    conv_a4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv_a4 = Dropout(0.2)(conv_a4)
    conv_a4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a4)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(conv_a4)
    print("conv_a4 = ", conv_a4.shape, "pool4 = ", pool4.shape)

    conv_a5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv_a5 = Dropout(0.2)(conv_a5)
    conv_a5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_a5)
    out1 = side_branch2(conv_a5, 16)
    print("conv_a5 = ", conv_a5.shape, "out1", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_a5)
    up1 = concatenate([conv_a4, up1], axis=1)
    conv_b4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv_b4 = Dropout(0.2)(conv_b4)
    conv_b4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b4)
    out2 = side_branch2(conv_b4, 8)
    print("up1 = ", up1.shape, "conv_b4 = ", conv_b4.shape, "out2 = ", out2.shape)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_b4)
    up2 = concatenate([conv_a3, up2], axis=1)
    conv_b3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv_b3 = Dropout(0.2)(conv_b3)
    conv_b3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b3)
    out3 = side_branch2(conv_b3, 4)
    print("up2 = ", up2.shape, "conv_b3 = ", conv_b3.shape, "out3 = ", out3.shape)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_b3)
    up3 = concatenate([conv_a2, up3], axis=1)
    conv_b2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv_b2 = Dropout(0.2)(conv_b2)
    conv_b2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b2)
    out4 = side_branch2(conv_b2, 2)
    print("up3 = ", up3.shape, "conv_b2 = ", conv_b2.shape, "out4 = ", out4.shape)
    #
    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv_b2)
    up4 = concatenate([conv_a1, up4], axis=1)
    conv_b1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv_b1 = Dropout(0.2)(conv_b1)
    conv_b1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv_b1)
    out5 = side_branch2(conv_b1, 1)
    print("up4 = ", up4.shape, "conv_b1 = ", conv_b1.shape, "out5 = ", out5.shape)
    #
    fuse = concatenate([out1, out2, out3, out4, out5], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    out4 = core.Permute((2, 3, 1))(out4)
    out5 = core.Permute((2, 3, 1))(out5)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    out4 = core.Activation('softmax', name='o4')(out4)
    out5 = core.Activation('softmax', name='o5')(out5)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    out4 = core.Permute((3, 1, 2), name='oo4')(out4)
    out5 = core.Permute((3, 1, 2), name='oo5')(out5)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "out4 = ", out4.shape, "out5 = ",
          out5.shape, "fuse = ",
          fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, out4, out5, fuse])
    loss = cross_entropy_balanced
    # loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'oo4': loss,
                                         'oo5': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet_all(n_ch, patch_height, patch_width):
    print("=====Using unet_all =====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    print("inputs = ", inputs.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 4)
    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(up1)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 2)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2, up3], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet4(n_ch, patch_height, patch_width):
    print("=====Using unet4(1 loss)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 4)
    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 2)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    fuse = core.Permute((2, 3, 1))(fuse)
    #
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'last': loss, }, metrics=['accuracy'])

    return model


def get_unet5(n_ch, patch_height, patch_width):
    print("=====Using unet5(3 losses)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 4)
    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 2)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='last')(out3)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_br(n_ch, patch_height, patch_width):
    print("=====Using unet_br (the output for br)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "pool1 = ", pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch3(conv3, 4)
    print("conv3 = ", conv3.shape, "out1", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch3(conv4, 2)
    print("up1 = ", up1.shape, "conv4 = ", conv4.shape, "out2 = ", out2.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch3(conv5, 1)
    print("up2 = ", up2.shape, "conv5 = ", conv5.shape, "out3 = ", out3.shape)
    #

    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    out1 = Boundary_Refinement2(out1)
    out2 = Boundary_Refinement2(out2)
    out3 = Boundary_Refinement2(out3)
    fuse = Boundary_Refinement2(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "fuse = ", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet_brnew(n_ch, patch_height, patch_width):
    print("=====Using unet_brnew (the output for br)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "pool1 = ", pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch3(conv3, 4)
    print("conv3 = ", conv3.shape, "out1", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch3(conv4, 2)
    print("up1 = ", up1.shape, "conv4 = ", conv4.shape, "out2 = ", out2.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch3(conv5, 1)
    print("up2 = ", up2.shape, "conv5 = ", conv5.shape, "out3 = ", out3.shape)
    #

    out1 = Boundary_Refinement2(out1)
    out2 = Boundary_Refinement2(out2)
    out3 = Boundary_Refinement2(out3)
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    fuse = Boundary_Refinement2(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "fuse = ", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet_br2(n_ch, patch_height, patch_width):
    print("=====Using unet_br2 (the last for br)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "pool1 = ", pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch3(conv3, 4)
    print("conv3 = ", conv3.shape, "out1", out1.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch3(conv4, 2)
    print("up1 = ", up1.shape, "conv4 = ", conv4.shape, "out2 = ", out2.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch3(conv5, 1)
    print("up2 = ", up2.shape, "conv5 = ", conv5.shape, "out3 = ", out3.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)
    print("fuse = ", fuse.shape)

    fuse = Boundary_Refinement2(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1 = ", out1.shape, "out2 = ", out2.shape, "out3 = ", out3.shape, "fuse = ", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, loss_weights=loss_w, metrics=['accuracy'])

    return model


def get_unet_dsm(n_ch, patch_height, patch_width):
    print("=====Using unet3 (differ score map)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    # conv1 = Lambda(slice)(conv1)
    # # Lambda(lambda conv1: conv1[:, 0:1, :, :])
    # print(conv1.shape)
    # i = input()
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 4)
    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 2)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)

    out1_0 = Lambda(slice_0)(out1)
    out1_1 = Lambda(slice_1)(out1)
    out2_0 = Lambda(slice_0)(out2)
    out2_1 = Lambda(slice_1)(out2)
    out3_0 = Lambda(slice_0)(out3)
    out3_1 = Lambda(slice_1)(out3)

    fuse_0 = concatenate([out1_0, out2_0, out3_0], axis=1)
    fuse_1 = concatenate([out1_1, out2_1, out3_1], axis=1)

    fuse_0 = Conv2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse_0)
    fuse_1 = Conv2D(1, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse_1)
    fuse = concatenate([fuse_0, fuse_1], axis=1)

    fuse = core.Permute((2, 3, 1))(fuse)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("out1.shape:", out1.shape)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_dm(n_ch, patch_height, patch_width):
    print("=====Using unet3(differ mask)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = Conv2D(2, (1, 1), data_format='channels_first')(conv3)

    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    out1_up = UpSampling2D(size=(2, 2), data_format='channels_first')(up1)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = Conv2D(2, (1, 1), data_format='channels_first')(conv4)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    out2_up = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, out2_up], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = Conv2D(2, (1, 1), data_format='channels_first')(conv5)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1_up, out2_up, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_dm2(n_ch, patch_height, patch_width):
    print("=====Using unet_dm(br information)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    print("conv1 = ", conv1, "pool1 = ", pool1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "pool2 = ", pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    out1 = Conv2D(2, (1, 1), data_format='channels_first')(conv3)

    print("conv3 = ", conv3.shape)
    #
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    out1_up = UpSampling2D(size=(2, 2), data_format='channels_first')(up1)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    out2 = Conv2D(2, (1, 1), data_format='channels_first')(conv4)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    out2_up = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, out2_up], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = Conv2D(2, (1, 1), data_format='channels_first')(conv5)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1_up, out2_up, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    out1 = Boundary_Refinement2(out1)
    out2 = Boundary_Refinement2(out2)
    out3 = Boundary_Refinement2(out3)
    fuse = Boundary_Refinement2(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_atrous(n_ch, patch_height, patch_width):
    print("=====Using unet3 (atrous_conv)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    atrous1 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                         data_format='channels_first')(conv1)
    print("conv1 = ", conv1.shape, "atrous1 = ", atrous1.shape)
    cont1 = concatenate([conv1, atrous1], axis=1)
    out1 = Conv2D(2, (1, 1), data_format="channels_first")(cont1)

    #
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(atrous1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    atrous2 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=3, activation='relu', padding='same',
                                         data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape, "atrous2 = ", atrous2.shape)
    cont2 = concatenate([conv2, atrous2], axis=1)
    out2 = Conv2D(2, (1, 1), data_format="channels_first")(cont2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(atrous2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    atrous3 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=4, activation='relu', padding='same',
                                         data_format='channels_first')(conv1)
    print("conv3 = ", conv3.shape)
    cont3 = concatenate([conv3, atrous3], axis=1)
    out3 = Conv2D(2, (1, 1), data_format="channels_first")(cont3)

    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_atrous2(n_ch, patch_height, patch_width):
    print("=====Using unet3 (atrous2)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    print("conv1 = ", conv1)
    #
    conv2 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv2)
    # pool2 = MaxPooling2D((2, 2),data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape)
    #
    conv3 = convolutional.AtrousConv2D(64, (2, 2), atrous_rate=4, activation='relu', padding='same',
                                       data_format='channels_first')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = convolutional.AtrousConv2D(64, (2, 2), atrous_rate=4, activation='relu', padding='same',
                                       data_format='channels_first')(conv3)
    out1 = side_branch2(conv3, 1)
    print("conv3 = ", conv3.shape)
    #
    up1 = concatenate([conv2, conv3], axis=1)
    conv4 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv4)
    out2 = side_branch2(conv4, 1)
    print("up1 = ", up1, "conv4 = ", conv4.shape)
    #
    up2 = concatenate([conv1, conv4], axis=1)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    out3 = side_branch2(conv5, 1)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), data_format='channels_first')(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_unet_atrous3(n_ch, patch_height, patch_width):
    print("=====Using unet3 (atrous3)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    print("conv1 = ", conv1)
    #
    conv2 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv2)
    # pool2 = MaxPooling2D((2, 2),data_format='channels_first')(conv2)
    print("conv2 = ", conv2.shape)
    #
    conv3 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=4, activation='relu', padding='same',
                                       data_format='channels_first')(conv2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = convolutional.AtrousConv2D(32, (2, 2), atrous_rate=4, activation='relu', padding='same',
                                       data_format='channels_first')(conv3)
    out1 = Conv2D(2, (1, 1), data_format='channels_first')(conv3)
    print("conv3 = ", conv3.shape)
    #
    conv4 = convolutional.AtrousConv2D(16, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = convolutional.AtrousConv2D(16, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                       data_format='channels_first')(conv4)
    up1 = concatenate([conv2, conv4], axis=1)
    out2 = Conv2D(2, (1, 1), data_format='channels_first')(up1)
    print("conv4 = ", conv4.shape)
    #

    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

    up2 = concatenate([conv1, conv5], axis=1)
    out3 = Conv2D(2, (1, 1), data_format='channels_first')(up2)
    print("up2 = ", up2, "conv5 = ", conv5.shape)
    #
    fuse = concatenate([out1, out2, out3], axis=1)
    fuse = Conv2D(2, (1, 1), data_format='channels_first')(fuse)

    out1 = core.Permute((2, 3, 1))(out1)
    out2 = core.Permute((2, 3, 1))(out2)
    out3 = core.Permute((2, 3, 1))(out3)
    fuse = core.Permute((2, 3, 1))(fuse)
    #
    out1 = core.Activation('softmax', name='o1')(out1)
    out2 = core.Activation('softmax', name='o2')(out2)
    out3 = core.Activation('softmax', name='o3')(out3)
    fuse = core.Activation('softmax', name='ofuse')(fuse)
    #
    out1 = core.Permute((3, 1, 2), name='oo1')(out1)
    out2 = core.Permute((3, 1, 2), name='oo2')(out2)
    out3 = core.Permute((3, 1, 2), name='oo3')(out3)
    fuse = core.Permute((3, 1, 2), name='last')(fuse)

    print("fuse shape:", fuse.shape)
    # # model
    model = Model(inputs=[inputs], outputs=[out1, out2, out3, fuse])
    loss = 'categorical_crossentropy'
    model.compile(optimizer='sgd', loss={'oo1': loss,
                                         'oo2': loss,
                                         'oo3': loss,
                                         'last': loss, }, metrics=['accuracy'])

    return model


def get_dianet(n_ch, patch_height, patch_width):
    print("=====Using dianet instead pooling=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)

    atrous1 = convolutional.AtrousConv2D(64, (2, 2), atrous_rate=2, activation='relu', padding='same',
                                         data_format='channels_first')(conv1)

    # filter1 = tf.constant(value=1, shape=[2, 2, 32, 32], dtype=tf.float32)
    # atrous1 = tf.nn.atrous_conv2d(value=conv1, filters=filter1 ,rate = 2,padding='SAME',name='atrous1')
    print("conv1 = ", conv1.shape, "atrous1 = ", atrous1.shape)

    conc = concatenate([conv1, atrous1], axis=1)
    # my_concat = Lambda(lambda x: concatenate([x[0], x[1]], axis=1))
    # conc = my_concat([conv1, atrous1])

    #
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='valid', data_format='channels_first')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='valid', data_format='channels_first')(conv2)

    atrous2 = convolutional.AtrousConv2D(64, (3, 3), atrous_rate=4, activation='relu', padding='valid',
                                         data_format='channels_first')(conv2)

    # filter2 = tf.constant(value=1, shape=[3, 3, 64, 64], dtype=tf.float32)
    # atrous2 = tf.nn.atrous_conv2d(value=conv2, filters=filter2, rate=4, padding='VALID', name='atrous2')
    print("conv2 = ", conv2.shape, "atrous2 = ", atrous2.shape)
    #
    conv3 = Conv2D(128, (5, 5), activation='relu', padding='valid', data_format='channels_first')(atrous2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (5, 5), activation='relu', padding='valid', data_format='channels_first')(conv3)
    print("conv3 = ", conv3.shape)
    #
    up = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    print("conc = ", conc.shape, "up =", up.shape)

    up = concatenate([conc, up], axis=1)
    # up = my_concat([conc,up])
    print("up =", up.shape)

    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    print("up = ", up, "conv5 = ", conv4.shape)
    #
    conv5 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv5 = core.Reshape((2, patch_height * patch_width))(conv5)
    conv5 = core.Permute((2, 1))(conv5)
    print("conv5 = ", conv5.shape)
    ############
    conv6 = core.Activation('softmax')(conv5)
    print("conv6 = ", conv6.shape)

    model = Model(input=inputs, output=conv6)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_dianet2(n_ch, patch_height, patch_width):
    print("=====Using dianet2=====")

    active = 'sigmoid'
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(64, (3, 3), activation=active, padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation=active, padding='same', data_format='channels_first')(conv1)

    atrous1 = convolutional.AtrousConv2D(64, (3, 3), atrous_rate=2, activation=active, padding='same',
                                         data_format='channels_first')(conv1)
    atrous1 = side_branch(atrous1, 1)
    print("conv1 = ", conv1.shape, "atrous1 = ", atrous1.shape)

    #
    conv2 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv1)
    conv2 = Dropout(0.2)(conv2)
    atrous2 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=3, activation=active, padding='same',
                                         data_format='channels_first')(conv2)
    atrous2 = side_branch(atrous2, 1)

    print("conv2 = ", conv2.shape, "atrous2 = ", atrous2.shape)
    #
    conv3 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv2)
    conv3 = Dropout(0.2)(conv3)
    atrous3 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=4, activation=active, padding='same',
                                         data_format='channels_first')(conv3)
    atrous3 = side_branch(atrous3, 1)
    print("conv3 = ", conv3.shape, "atrous3 = ", atrous3.shape)

    conv4 = Conv2D(16, (3, 3), activation=active, padding='same', data_format='channels_first')(conv3)
    conv4 = Dropout(0.2)(conv4)
    atrous4 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=5, activation=active, padding='same',
                                         data_format='channels_first')(conv4)
    atrous4 = side_branch(atrous4, 1)
    print("conv4 = ", conv4.shape, "atrous4 = ", atrous4.shape)

    conv5 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv4)
    conv5 = Dropout(0.2)(conv5)
    atrous5 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=6, activation=active, padding='same',
                                         data_format='channels_first')(conv5)
    atrous5 = side_branch(atrous5, 1)
    print("conv5 = ", conv5.shape, "atrous5 = ", atrous5.shape)

    out = concatenate([conv1, atrous1, atrous2, atrous3, atrous4, atrous5], axis=1)
    out = Conv2D(1, (1, 1), activation=active, padding='same', data_format='channels_first')(out)
    print("out =", out.shape)

    model = Model(input=inputs, output=out)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_dianet3(n_ch, patch_height, patch_width):
    print("=====Using dianet3=====")
    active = 'sigmoid'
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(64, (3, 3), activation=active, padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation=active, padding='same', data_format='channels_first')(conv1)

    atrous1 = convolutional.AtrousConv2D(64, (3, 3), atrous_rate=2, activation=active, padding='same',
                                         data_format='channels_first')(conv1)
    atrous1 = side_branch(atrous1, 1)

    print("conv1 = ", conv1.shape, "atrous1 = ", atrous1.shape)

    #
    conv2 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv1)
    conv2 = Dropout(0.2)(conv2)
    atrous2 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=3, activation=active, padding='same',
                                         data_format='channels_first')(conv2)
    atrous2 = side_branch(atrous2, 1)

    print("conv2 = ", conv2.shape, "atrous2 = ", atrous2.shape)
    #
    conv3 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv2)
    conv3 = Dropout(0.2)(conv3)
    atrous3 = convolutional.AtrousConv2D(32, (3, 3), atrous_rate=4, activation=active, padding='same',
                                         data_format='channels_first')(conv3)
    atrous3 = side_branch(atrous3, 1)
    print("conv3 = ", conv3.shape, "atrous3 = ", atrous3.shape)

    conv4 = Conv2D(16, (3, 3), activation=active, padding='same', data_format='channels_first')(conv3)
    conv4 = Dropout(0.2)(conv4)
    atrous4 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=5, activation=active, padding='same',
                                         data_format='channels_first')(conv4)
    atrous4 = side_branch(atrous4, 1)
    print("conv4 = ", conv4.shape, "atrous4 = ", atrous4.shape)

    conv5 = Conv2D(32, (3, 3), activation=active, padding='same', data_format='channels_first')(conv4)
    conv5 = Dropout(0.2)(conv5)
    atrous5 = convolutional.AtrousConv2D(16, (3, 3), atrous_rate=6, activation=active, padding='same',
                                         data_format='channels_first')(conv5)
    atrous5 = side_branch(atrous5, 1)
    print("conv5 = ", conv5.shape, "atrous5 = ", atrous5.shape)

    out = concatenate([conv1, atrous1, atrous2, atrous3, atrous4, atrous5], axis=1)
    out = Conv2D(1, (1, 1), activation=active, padding='same', data_format='channels_first')(out)
    print("out =", out.shape)

    fuse = Concatenate(axis=1)([atrous1, atrous2, atrous3, a])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, data_format='channels_first')(
        fuse)  # 480 480 1

    # outputs
    activ = "sigmoid"
    o1 = Activation(activ, name='o1')(atrous1)
    o2 = Activation(activ, name='o2')(atrous2)
    o3 = Activation(activ, name='o3')(atrous3)
    o4 = Activation(activ, name='o4')(atrous4)
    o5 = Activation(activ, name='o5')(atrous5)
    ofuse = Activation(activ, name='ofuse')(fuse)

    # model
    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])
    # model = Model(inputs = [img_input], outputs=[ofuse])
    # filepath = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # load_weights_from_hdf5_group_by_name(model, filepath)

    model.compile(loss={'o1': cross_entropy_balanced,
                        'o2': cross_entropy_balanced,
                        'o3': cross_entropy_balanced,
                        'o4': cross_entropy_balanced,
                        'o5': cross_entropy_balanced,
                        'ofuse': cross_entropy_balanced,
                        },
                  metrics={'ofuse': ofuse_pixel_error},
                  optimizer='adam')

    return model


# def get_hed(n_ch, patch_height, patch_width):
#     print("=====Using hed net=====")
#
#     img_input = Input(shape=(n_ch, patch_height, patch_width), name='input')
#     # Block 1
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv1')(
#         img_input)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv2')(x)
#     print(x.name, x.shape, end="  ")
#     b1 = side_branch(x, 1)  # 480 480 1
#     x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block1_pool')(
#         x)  # 240 240 64
#     print(x.name, x.shape)
#
#     # Block 2
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block2_conv1')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block2_conv2')(x)
#     print(x.name, x.shape, end="  ")
#     b2 = side_branch(x, 2)  # 480 480 1
#     x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block2_pool')(
#         x)  # 120 120 128
#     print(x.name, x.shape)
#
#     # Block 3
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv1')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv2')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv3')(x)
#     print(x.name, x.shape, end="  ")
#     b3 = side_branch(x, 4)  # 480 480 1
#     x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block3_pool')(
#         x)  # 60 60 256
#     print(x.name, x.shape)
#
#     # Block 4
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block4_conv1')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block4_conv2')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block4_conv3')(x)
#     print(x.name, x.shape, end="  ")
#     b4 = side_branch(x, 8)  # 480 480 1
#     x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block4_pool')(
#         x)  # 30 30 512
#     print(x.name, x.shape)
#
#     # Block 5
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block5_conv1')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block5_conv2')(x)
#     print(x.name, x.shape, end="  ")
#     x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block5_conv3')(
#         x)  # 30 30 512
#     print(x.name, x.shape)
#     b5 = side_branch(x, 16)  # 480 480 1
#
#     # fuse
#     fuse = Concatenate(axis=1)([b1, b2, b3, b4, b5])
#     fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, data_format='channels_first')(
#         fuse)  # 480 480 1
#
#     # outputs
#     activ = "sigmoid"
#     o1 = Activation(activ, name='o1')(b1)
#     o2 = Activation(activ, name='o2')(b2)
#     o3 = Activation(activ, name='o3')(b3)
#     o4 = Activation(activ, name='o4')(b4)
#     o5 = Activation(activ, name='o5')(b5)
#     ofuse = Activation(activ, name='ofuse')(fuse)
#
#     # model
#     model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
#     # model = Model(inputs = [img_input], outputs=[ofuse])
#     # filepath = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     # load_weights_from_hdf5_group_by_name(model, filepath)
#
#     model.compile(loss={'o1': cross_entropy_balanced,
#                         'o2': cross_entropy_balanced,
#                         'o3': cross_entropy_balanced,
#                         'o4': cross_entropy_balanced,
#                         'o5': cross_entropy_balanced,
#                         'ofuse': cross_entropy_balanced,
#                         },
#                   metrics={'ofuse': ofuse_pixel_error},
#                   optimizer='adam')
#
#     return model

def get_hed(n_ch, patch_height, patch_width):
    print("=====Using hed net=====")

    img_input = Input(shape=(n_ch, patch_height, patch_width), name='input')
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv1')(
        img_input)
    print(x.name, x.shape, end="  ")
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv2')(x)
    print(x.name, x.shape, end="  ")
    b1 = side_branch(x, 1)  # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block1_pool')(
        x)  # 240 240 64
    print(x.name, x.shape)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block2_conv1')(x)
    print(x.name, x.shape, end="  ")
    x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block2_conv2')(x)
    print(x.name, x.shape, end="  ")
    b2 = side_branch(x, 2)  # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', data_format='channels_first', name='block2_pool')(
        x)  # 120 120 128
    print(x.name, x.shape)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv1')(x)
    print(x.name, x.shape, end="  ")
    x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv2')(x)
    print(x.name, x.shape, end="  ")
    x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block3_conv3')(x)
    print(x.name, x.shape, end="  ")
    b3 = side_branch(x, 4)  # 480 480 1
    print(x.name, x.shape)

    # fuse
    fuse = Concatenate(axis=1)([b1, b2, b3])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, data_format='channels_first')(
        fuse)  # 480 480 1

    # outputs
    activ = "sigmoid"
    o1 = Activation(activ, name='o1')(b1)
    o2 = Activation(activ, name='o2')(b2)
    o3 = Activation(activ, name='o3')(b3)
    ofuse = Activation(activ, name='ofuse')(fuse)

    # model
    model = Model(inputs=[img_input], outputs=[o1, o2, o3, ofuse])
    # model = Model(inputs = [img_input], outputs=[ofuse])
    # filepath = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # load_weights_from_hdf5_group_by_name(model, filepath)

    model.compile(loss={'o1': cross_entropy_balanced,
                        'o2': cross_entropy_balanced,
                        'o3': cross_entropy_balanced,
                        'ofuse': cross_entropy_balanced,
                        },
                  metrics={'ofuse': ofuse_pixel_error},
                  optimizer='adam')

    return model


def get_hed2(n_ch, patch_height, patch_width):
    # Input
    img_input = Input(shape=(n_ch, patch_height, patch_width), name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv1')(
        img_input)
    print(x.name, x.shape, end="  ")
    b1 = side_branch(x, 1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv2')(x)
    print(x.name, x.shape, end="  ")
    b2 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv3')(x)
    print(x.name, x.shape, end="  ")
    b3 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv')(x)
    print(x.name, x.shape, end="  ")
    b4 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv5')(x)
    print(x.name, x.shape, end="  ")
    b5 = side_branch(x, 1)  # 480 480 1

    # fuse
    fuse = Concatenate(axis=1)([b1, b2, b3, b4, b5])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, data_format='channels_first')(
        fuse)  # 480 480 1

    # outputs
    activ = "sigmoid"
    o1 = Activation(activ, name='o1')(b1)
    o2 = Activation(activ, name='o2')(b2)
    o3 = Activation(activ, name='o3')(b3)
    o4 = Activation(activ, name='o4')(b4)
    o5 = Activation(activ, name='o5')(b5)
    ofuse = Activation(activ, name='ofuse')(fuse)

    # model
    model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
    # model = Model(inputs = [img_input], outputs=[ofuse])
    # filepath = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # load_weights_from_hdf5_group_by_name(model, filepath)

    model.compile(loss={'o1': cross_entropy_balanced,
                        'o2': cross_entropy_balanced,
                        'o3': cross_entropy_balanced,
                        'o4': cross_entropy_balanced,
                        'o5': cross_entropy_balanced,
                        'ofuse': cross_entropy_balanced,
                        },
                  metrics={'ofuse': ofuse_pixel_error},
                  optimizer='adam')

    return model


def get_hed3(n_ch, patch_height, patch_width):
    print("=====Using hed3=====")
    img_input = Input(shape=(n_ch, patch_height, patch_width), name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv1')(
        img_input)
    print(x.name, x.shape, end="  ")
    b1 = side_branch(x, 1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv2')(x)
    print(x.name, x.shape, end="  ")
    b2 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv3')(x)
    print(x.name, x.shape, end="  ")
    b3 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv')(x)
    print(x.name, x.shape, end="  ")
    b4 = side_branch(x, 1)  # 480 480 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first', name='block1_conv5')(x)
    print(x.name, x.shape, end="  ")
    b5 = side_branch(x, 1)  # 480 480 1

    # fuse
    fuse = Concatenate(axis=1)([b1, b2, b3, b4, b5])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, data_format='channels_first')(
        fuse)  # 480 480 1

    # outputs
    activ = "sigmoid"
    ofuse = Activation(activ, name='ofuse')(fuse)

    # model
    model = Model(inputs=[img_input], outputs=[ofuse])
    # model = Model(inputs = [img_input], outputs=[ofuse])
    # filepath = './models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # load_weights_from_hdf5_group_by_name(model, filepath)

    model.compile(loss={'ofuse': cross_entropy_balanced,
                        },
                  metrics={'ofuse': ofuse_pixel_error},
                  optimizer='adam')

    return model


def get_test(n_ch, patch_height, patch_width):
    print("=====Using unet_br (softmax)=====")
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    print("conv1 = ", conv1.shape)
    conv1 = Boundary_Refinement(conv1)

    model = Model(inputs=[inputs], outputs=conv1)
    loss = cross_entropy_balanced
    model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])

    return model

# # # #
# model = get_hed(1, 48, 48)
# print("Check: final output of the network:")
# print(model.output_shape)

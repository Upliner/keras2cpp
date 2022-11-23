import struct
from functools import singledispatch

import numpy as np
import keras

from keras.layers import (
    Layer,
    Dense,
    Conv1D, Conv2D,
    Conv2DTranspose,
    LocallyConnected1D, LocallyConnected2D,
    Flatten,
    ELU,
    Activation,
    MaxPooling2D,
    UpSampling2D,
    LSTM,
    Embedding,
    BatchNormalization,
    InputLayer,
    Add, Multiply, Concatenate,
    )

LAYERS = {
    Dense : 1,
    Conv1D : 2,
    Conv2D : 3,
    Conv2DTranspose : 3,
    LocallyConnected1D : 4,
    LocallyConnected2D : 5,
    Flatten : 6,
    ELU : 7,
    Activation : 8,
    MaxPooling2D : 9,
    LSTM : 10,
    Embedding : 11,
    BatchNormalization : 12,
    UpSampling2D : 13,

    Add : 101,
    Multiply : 102,
    Concatenate : 103,

    InputLayer : 1000
}

ACTIVATIONS = (
    'linear',
    'relu',
    'elu',
    'softplus',
    'softsign',
    'sigmoid',
    'tanh',
    'hard_sigmoid',
    'softmax',
)

PADDINGS = {
    'valid' : 0,
    'same' : 1,
}

ENDIAN = '='


def input_count(layer_id):
    if 100<=layer_id<200:
        return 2;

    if 1000<=layer_id<2000:
        return 0;

    return 1


def write_tensor(f, data, dims=1):
    """
    Writes tensor as flat array of floats to file in 1024 chunks,
    prevents memory explosion writing very large arrays to disk
    when calling struct.pack().
    """
    for stride in data.shape[:dims]:
        f.write(struct.pack(f'{ENDIAN}I', stride))

    data = data.flatten()
    step = 1024
    written = 0

    for i in np.arange(0, len(data), step):
        remaining = min(len(data) - i, step)
        written += remaining
        f.write(struct.pack(f'{ENDIAN}{remaining}f', *data[i: i + remaining]))

    assert written == len(data)


def export_activation(activation, f):
    try:
        f.write(struct.pack(f'{ENDIAN}I', ACTIVATIONS.index(activation) + 1))
    except ValueError as exc:
        raise NotImplementedError(activation) from exc


@singledispatch
def export(layer, _):
    raise NotImplementedError(layer)


def empty_exporter(_layer, _f):
    pass


for typ in (Flatten, InputLayer, Add, Multiply, Concatenate):
    export.register(typ, empty_exporter)


@export.register(Activation)
def _(layer, f):
    activation = layer.get_config()['activation']
    export_activation(activation, f)


@export.register(ELU)
def _(layer, f):
    f.write(struct.pack(f'{ENDIAN}f', layer.alpha))


@export.register(BatchNormalization)
def _(layer, f):
    epsilon = layer.epsilon
    gamma = layer.get_weights()[0]
    beta = layer.get_weights()[1]
    pop_mean = layer.get_weights()[2]
    pop_variance = layer.get_weights()[3]

    weights = gamma / np.sqrt(pop_variance + epsilon)
    biases = beta - pop_mean * weights

    write_tensor(f, weights)
    write_tensor(f, biases)


@export.register(Dense)
def _(layer, f):
    # shape: (outputs, dims)
    weights = layer.get_weights()[0].transpose()
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    write_tensor(f, weights, 2)
    write_tensor(f, biases)
    export_activation(activation, f)


@export.register(Conv1D)
def _(layer, f):
    # shape: (outputs, steps, dims)
    weights = layer.get_weights()[0].transpose(2, 0, 1)
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    write_tensor(f, weights, 3)
    write_tensor(f, biases)
    export_activation(activation, f)


def export_conv2d(layer, f):
    # shape: (outputs, rows, cols, depth)
    weights = layer.get_weights()[0].transpose(3, 0, 1, 2)
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    f.write(struct.pack(f'{ENDIAN}I', PADDINGS[layer.padding]))
    strides = layer.strides
    if isinstance(layer, Conv2DTranspose) and strides != (1, 1):
        raise Exception("Unsupported Conv2DTranspose layer")
    f.write(struct.pack(f'{ENDIAN}II', *strides))

    write_tensor(f, weights, 4)
    write_tensor(f, biases)
    export_activation(activation, f)

export.register(Conv2D, export_conv2d);
export.register(Conv2DTranspose, export_conv2d);

@export.register(LocallyConnected1D)
def _(layer, f):
    # shape: (new_steps, outputs, ksize*dims)
    weights = layer.get_weights()[0].transpose(0, 2, 1)
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    write_tensor(f, weights, 3)
    write_tensor(f, biases, 2)
    export_activation(activation, f)


@export.register(LocallyConnected2D)
def _(layer, f):
    # shape: (rows*cols, outputs, ksize*depth)
    weights = layer.get_weights()[0]
    # weights = weights.transpose(0, 2, 1)
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    write_tensor(f, weights, 3)
    write_tensor(f, biases, 2)
    export_activation(activation, f)


@export.register(MaxPooling2D)
def _(layer, f):
    pool_size = layer.get_config()['pool_size']

    f.write(struct.pack(f'{ENDIAN}II', *pool_size))


@export.register(UpSampling2D)
def _(layer, f):
    size = layer.get_config()['size']

    f.write(struct.pack(f'{ENDIAN}II', *size))


@export.register(LSTM)
def _(layer, f):
    inner_activation = layer.get_config()['recurrent_activation']
    activation = layer.get_config()['activation']
    return_sequences = int(layer.get_config()['return_sequences'])

    weights = layer.get_weights()
    units = layer.units

    kernel, rkernel, bias = ([x[i: i+units] for i in range(0, 4*units, units)]
                             for x in (weights[0].transpose(),
                                       weights[1].transpose(),
                                       weights[2]))
    bias = [x.reshape(1, -1) for x in bias]
    for tensors in zip(kernel, rkernel, bias):
        for tensor in tensors:
            write_tensor(f, tensor, 2)

    export_activation(inner_activation, f)
    export_activation(activation, f)
    f.write(struct.pack(f'{ENDIAN}I', return_sequences))


@export.register(Embedding)
def _(layer, f):
    weights = layer.get_weights()[0]
    write_tensor(f, weights, 2)


def export_repeat(layer, f):
    f.write(struct.pack(f'{ENDIAN}I', layer.repnum))


def _aslist(item):
    return item if isinstance(item, list) else (item,)


def layer_type_id(layer) -> int:
    return LAYERS[type(layer)]


def export_model(model: keras.models.Model, filename: str):
    with open(filename, 'wb') as f:
        layers = [layer for layer in model.layers
                  if type(layer).__name__ not in ['Dropout']]

        if not isinstance(layers[0], InputLayer):
            layers = [layers[0].inbound_nodes[0].inbound_layers] + layers

        f.write(struct.pack(f'{ENDIAN}I', len(layers)))

        def get_inputs(layer: Layer) -> list[int]:
            return [
                layers.index(in_layer)
                for node in layer.inbound_nodes
                for in_layer in _aslist(node.inbound_layers)
            ]

        def export_layer(layer: Layer, inputs=None):
            type_id = layer_type_id(layer)
            f.write(struct.pack(f'{ENDIAN}I', type_id))

            if inputs is None:
                inputs = get_inputs(layer)
            expected_inputs = input_count(type_id)
            assert len(inputs) == expected_inputs
            export(layer, f)
            for input in inputs:
                f.write(struct.pack(f'{ENDIAN}I', input))

        for layer in layers:
            export_layer(layer)

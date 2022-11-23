import os
import sys
import pprint
import re

import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense,
    ELU, Embedding, Flatten, LSTM, MaxPooling2D, UpSampling2D
)

from keras2cpp import export_model

np.set_printoptions(precision=25, threshold=sys.maxsize)


os.makedirs('test', exist_ok=True)
for path_ in os.listdir('test'):
    os.remove('test/' + path_)

os.makedirs('models', exist_ok=True)
for path_ in os.listdir('models'):
    os.remove('models/' + path_)


def c_array(a):
    s = pprint.pformat(a.flatten())

    s = re.sub(r'[ \t\n]*', '', s)
    s = re.sub(r'[ \t]*,[ \t]*', ', ', s)
    s = re.sub(r'[ \t]*\][, \t]*', '} ', s)
    s = re.sub(r'[ \t]*\[[ \t]*', '{', s)
    s = s.replace('array(', '').replace(')', '')
    s = re.sub(r'[, \t]*dtype=float32', '', s)
    s = s.strip()

    shape = ''
    if a.shape:
        shape = repr(a.shape)
        shape = re.sub(r',*\)', '}', shape.replace('(', '{'))
    else:
        shape = '{1}'
    return shape, s


TEST_CASE = '''/* Autogenerated file, DO NOT EDIT */
#pragma once

#include "src/model.h"

namespace test {
    inline auto %(name)s() {
        printf("TEST %(name)s\\n");
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
        keras2cpp::Tensor in%(x_shape)s;
        in.data_ = %(x_data)s;
        keras2cpp::Tensor target%(y_shape)s;
        target.data_ = %(y_data)s;
    #pragma GCC diagnostic pop
        auto [model, load_time] = keras2cpp::timeit(keras2cpp::Model::load, "%(path)s");
        auto [output, apply_time] = keras2cpp::timeit(model, in);
        for (size_t i = 0; i < target.dims_[0]; ++i)
            kassert_eq(target(i), output(i), %(eps)s);
        return std::make_tuple(load_time, apply_time);
    }
}
'''


def output_testcase(model, test_x, test_y, name, eps):
    print(f'Processing {name}')
    model.compile(loss='mse', optimizer='adam')
    model.fit(test_x, test_y, epochs=1, verbose=False)
    predict_y = model.predict(test_x).astype('f')
    print(model.summary())

    path = os.path.abspath(f'models/{name}.model')
    export_model(model, path)

    with open(f'test/{name}.h', 'w') as f:
        x_shape, x_data = c_array(test_x[0])
        y_shape, y_data = c_array(predict_y[0])

        f.write(TEST_CASE % dict(name=name, path=path, eps=eps,
                                 x_shape=x_shape,
                                 x_data=x_data,
                                 y_shape=y_shape,
                                 y_data=y_data))


# Dense 1x1
test_x = np.arange(10)
test_y = test_x * 10 + 1
model = Sequential([
    Dense(1, input_dim=1)
])
output_testcase(model, test_x, test_y, 'dense_1x1', '1e-6')


# Dense 10x1
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(1, input_dim=10)
])
output_testcase(model, test_x, test_y, 'dense_10x1', '1e-6')


# Dense 2x2
test_x = np.random.rand(10, 2).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(2, input_dim=2),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'dense_2x2', '1e-6')


# Dense 10x10
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'dense_10x10', '1e-6')


# Dense 10x10x10
test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Dense(10)
])
output_testcase(model, test_x, test_y, 'dense_10x10x10', '1e-6')


# Conv 2x2
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_2x2', '1e-6')


# Conv 3x3
test_x = np.random.rand(10, 3, 3, 1).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (3, 3), input_shape=(3, 3, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3', '1e-6')


# Conv 3x3x3
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(3, (3, 3), input_shape=(10, 10, 3)),
    Flatten(),
    BatchNormalization(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3x3', '1e-6')


# Conv 2x2 with padding
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), padding="same"),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_2x2_padding', '1e-6')


# Conv 3x3 with padding
test_x = np.random.rand(10, 3, 3, 1).astype('f').astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (3, 3), input_shape=(3, 3, 1), padding="same"),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3_padding', '1e-6')


# Conv 3x3x3 with padding
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(3, (3, 3), input_shape=(10, 10, 3), padding="same"),
    Flatten(),
    BatchNormalization(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_3x3x3_padding', '1e-6')


# Activation ELU
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 1).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    ELU(alpha=0.5),
    Dense(1, activation='elu')
])
output_testcase(model, test_x, test_y, 'elu_10', '1e-6')


# Activation relu
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10),
    Activation('relu')
])
output_testcase(model, test_x, test_y, 'relu_10', '1e-6')


# Dense relu
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='relu'),
    Dense(10, input_dim=10, activation='relu'),
    Dense(10, input_dim=10, activation='relu')
])
output_testcase(model, test_x, test_y, 'dense_relu_10', '1e-6')


# Dense relu
test_x = np.random.rand(1, 10).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Dense(10, input_dim=10, activation='tanh'),
    Dense(10, input_dim=10, activation='tanh'),
    Dense(10, input_dim=10, activation='tanh')
])
output_testcase(model, test_x, test_y, 'dense_tanh_10', '1e-6')


# Conv softplus
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='softplus'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_softplus_2x2', '1e-6')


# Conv hardsigmoid
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='hard_sigmoid'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_hard_sigmoid_2x2', '1e-6')


# Conv sigmoid
test_x = np.random.rand(10, 2, 2, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    Conv2D(1, (2, 2), input_shape=(2, 2, 1), activation='sigmoid'),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'conv_sigmoid_2x2', '1e-6')


# Maxpooling2D 1x1
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(1, 1), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_1x1', '1e-6')


# Maxpooling2D 2x2
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(2, 2), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_2x2', '1e-6')


# Maxpooling2D 3x2x2
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(2, 2), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_3x2x2', '1e-6')


# Maxpooling2D 3x3x3
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    MaxPooling2D(pool_size=(3, 3), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'maxpool2d_3x3x3', '1e-6')


# UpSampling2D 1x1
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    UpSampling2D(size=(1, 1), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'up_sampling2d_1x1', '1e-6')


# UpSampling2D 2x2
test_x = np.random.rand(10, 10, 10, 1).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    UpSampling2D(size=(2, 2), input_shape=(10, 10, 1)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'up_sampling2d_2x2', '1e-6')


# UpSampling2D 3x2x2
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    UpSampling2D(size=(2, 2), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'up_sampling2d_3x2x2', '1e-6')


# UpSampling2D 3x3x3
test_x = np.random.rand(10, 10, 10, 3).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    UpSampling2D(size=(3, 3), input_shape=(10, 10, 3)),
    Flatten(),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'up_sampling2d_3x3x3', '1e-6')


# LSTM simple 7x20
test_x = np.random.rand(10, 7, 20).astype('f')
test_y = np.random.rand(10, 3).astype('f')
model = Sequential([
    LSTM(3, return_sequences=False, input_shape=(7, 20))
])
output_testcase(model, test_x, test_y, 'lstm_simple_7x20', '1e-6')


# LSTM simple stacked 16x9
test_x = np.random.rand(10, 16, 9).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LSTM(16, return_sequences=False, input_shape=(16, 9)),
    Dense(3, input_dim=16, activation='tanh'),
    Dense(1)
])
output_testcase(model, test_x, test_y, 'lstm_simple_stacked_16x9', '1e-6')


# LSTM stacked 64x83
test_x = np.random.rand(10, 64, 83).astype('f')
test_y = np.random.rand(10, 1).astype('f')
model = Sequential([
    LSTM(16, return_sequences=True, input_shape=(64, 83)),
    LSTM(16, return_sequences=False),
    Dense(1, activation='sigmoid')
])
output_testcase(model, test_x, test_y, 'lstm_stacked_64x83', '1e-6')


# Embedding 64
np.random.seed(10)
test_x = np.random.randint(100, size=(32, 10)).astype('f')
test_y = np.random.rand(32, 20).astype('f')
model = Sequential([
    Embedding(100, 64, input_length=10),
    Flatten(),
    # Dropout(0.5),
    Dense(20, activation='sigmoid')
])
output_testcase(model, test_x, test_y, 'embedding_64', '1e-6')


# Benchmark
test_x = np.random.rand(1, 128, 128, 3).astype('f')
test_y = np.random.rand(1, 10).astype('f')
model = Sequential([
    Conv2D(16, (7, 7), input_shape=(128, 128, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    ELU(),
    Conv2D(8, (3, 3)),
    Flatten(),
    Dense(1000, activation='relu'),
    Dense(10)
])
output_testcase(model, test_x, test_y, 'benchmark', '1e-3')


os.system('clang-format -i --style=file test/*.h')

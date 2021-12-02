import tensorflow as tf
from pandas import Series
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def build():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    return model

def build_compile_fit_dataset(dataset, epochs=5, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    model = build()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(dataset, epochs=epochs)
    return (model, history)

def build_compile_fit(x, y, epochs=5, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    model = build()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(x=x, y=y, epochs=epochs)
    return (model, history)


def reshape_input(x, y, normalize=True):
    no_elements = x.shape[0]
    images = x.apply(Series).stack().to_numpy().reshape(no_elements, 28, 28, 1)
    if normalize:
        images /= 255
    return (images, y)

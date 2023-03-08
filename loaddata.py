from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def get_MNIST():
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x, xl), (y, yl) = get_MNIST()
    print(x.shape)

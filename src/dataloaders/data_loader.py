from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

def load_iris(batch_size=32):

    iris_data = fetch_data("iris")

    print(iris_data.describe())

    X, y = fetch_data("iris", return_X_y=True)

    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, shuffle=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_Train, y_Train))
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_Test, y_Test))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def load_mnist(batch_size = 64):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  x_train = np.reshape(x_train, (-1,28, 28, 1))
  x_test = np.reshape(x_test, (-1, 28, 28, 1))
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  # Reserve 10,000 samples for validation.
  x_val = x_train[10000:12000]
  y_val = y_train[10000:12000]
  x_train = x_train[:10000]
  y_train = y_train[:10000]

  # Prepare the training dataset.
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.batch(batch_size)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(batch_size)

  return train_dataset, val_dataset, test_dataset


def load_cifar10(batch_size = 64):
  (x_train, y_train), (x_test , y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  #x_val = x_train[10000:12000]
  #y_val = y_train[10000:12000]
  #x_train = x_train[:10000]
  #y_train = y_train[:10000]

  x_val = x_train[-10000:]
  y_val = y_train[-10000:]
  x_train = x_train[:-10000]
  y_train = y_train[:-10000]

  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.batch(batch_size)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(batch_size)
  return train_dataset, val_dataset, test_dataset

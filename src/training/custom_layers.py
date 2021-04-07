import tensorflow as tf
class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, padding,pool_size,activation,maxpool=False):
        super().__init__()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,padding=padding)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.bn = tf.keras.layers.BatchNormalization(axis=3)
        self.maxpoollayer = tf.keras.layers.MaxPool2D(pool_size=pool_size,strides=1,padding="same")
        self.maxpool = maxpool


    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.maxpoollayer(x)
        return x



class ConvAdjustChannels(tf.keras.Model):
    def __init__(self, C_out):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=C_out, kernel_size=1,padding="same")
        self.bn = tf.keras.layers.BatchNormalization(axis=3)



    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x)
        return x
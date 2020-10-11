import numpy as np
import tensorflow as tf

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class, cell_count=128):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(cell_count, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        B, K, N, D = input_images.shape
        images = tf.reshape(input_images, (-1, K*N, D))

        labels = tf.concat((input_labels[:, :-1],
                            tf.zeros_like(input_labels[:, -1:])), axis=1)
        labels = tf.reshape(labels, (-1, K*N, N))

        inp = tf.concat((images, labels), -1)
        out = self.layer1(inp)
        out = self.layer2(out)
        out = tf.reshape(out, (-1, K, N, N))
        #############################
        return out

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        loss = tf.compat.v1.losses.softmax_cross_entropy(labels[:, -1:, :, :],
                                                         preds[:, -1:, :, :])
        loss = tf.reduce_mean(loss)
        #############################

        return loss

class MANN2(MANN):

    def __init__(self, num_classes, samples_per_class, cell_count=128):
        super(MANN2, self).__init__(num_classes, samples_per_class, cell_count=cell_count)

        self.layer1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                             input_shape=(28, 28))
        self.layer2 = tf.keras.layers.MaxPooling2D((3, 3))
        self.layer3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.layer4 = tf.keras.layers.MaxPooling2D((3, 3))
        self.layer5 = tf.keras.layers.LSTM(cell_count, return_sequences=True)
        self.layer6 = tf.keras.layers.LSTM(cell_count, return_sequences=True)
        self.layer7 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        B, K, N, D = input_images.shape
        d = int(np.sqrt(D))

        images = tf.reshape(input_images, (B*K*N, D))
        images = tf.reshape(images, (B*K*N, d, d, 1))
        labels = tf.concat((input_labels[:, :-1],
                            tf.zeros_like(input_labels[:, -1:])), axis=1)
        labels = tf.reshape(labels, (-1, K*N, N))

        out = images
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)

        out = tf.reshape(out, (out.shape[0], -1))

        out = tf.reshape(out, (B,K*N,-1))
        out = tf.concat((out, labels), -1)
        
        out = self.layer5(out)
        # out = self.layer6(out)
        out = self.layer7(out)
        out = tf.reshape(out, (-1, K, N, N))
        #############################
        return out
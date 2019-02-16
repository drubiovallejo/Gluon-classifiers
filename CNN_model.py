"""
David Rubio Vallejo

Implements a CNN with an embedding layer consisting of the word vectors for the input words, a 2D convolutional layer
with MaxPooling and Dropout, and an output layer mapping into a single neuron.

A CNN model is typically represented as the four-tuple (N, C, H, W), where
N = Batch size
C = Channels (# of filters that the convolutional layer has)
H = Length of the sentences ('H'eight of the feature-vector matrix, the # of rows)
W = Length of each word-vector in a sentence ('W'idth of the feature-vector matrix, the # of columns)
"""

# coding: utf-8

import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock


class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=1, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            # Embedding layer
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            # 2D Convolutional layer
            self.conv1 = gluon.nn.Conv2D(channels=100, kernel_size=(3, emb_output_dim), activation='relu')
            # MaxPooling followed by dropout
            self.pool1 = gluon.nn.GlobalMaxPool2D()
            self.activation1 = gluon.nn.Dropout(0.2)
            # Output layer
            self.out = gluon.nn.Dense(num_classes)

    def hybrid_forward(self, F, data):
        # print('Data shape', data.shape)
        embedded = self.embedding(data)
        # print('Embedded shape',embedded.shape)
        x = self.conv1(embedded)
        x = self.pool1(x)
        x = self.activation1(x)
        x = self.out(x)
        return x

"""
David Rubio Vallejo

Implements an LSTM with an embedding layer consisting of the word vectors for the input words, the recurrent LSTM layer,
 and an output layer mapping into a single neuron.

An LSTM model in Gluon is represented as a triple (S, N, W), where
S = Length of the sequence
N = Batch size
W = Length of each word-vector for a token in a sentence.
"""

import mxnet.gluon as gluon
from mxnet.gluon import HybridBlock
from mxnet.ndarray import squeeze, swapaxes

class LSTMTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, num_classes=1, prefix=None, params=None):
        super(LSTMTextClassifier, self).__init__(prefix=prefix, params=params)

        with self.name_scope():

            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)

            self.lstm = gluon.rnn.LSTM(hidden_size=100)
            self.activation1 = gluon.nn.Dropout(0.2)
            self.out = gluon.nn.Dense(num_classes)

    def hybrid_forward(self, F, data):
        # data.shape is a triple of (16, 1, 64). Need to eliminate that redundant second dimension and transpose it
        # before attaching the embeddings
        data = squeeze(data)
        data = data.T
        embedded = self.embedding(data)

        x = self.lstm(embedded)
        x - self.activation1(x)

        # Swap the first and second axes to bring it from (length, batch size, width) to (batch size, length, width),
        # before passing it to the outer layer (only recurrent layers use the first ordering).
        x = swapaxes(x, 0, 1)

        x = self.out(x)

        return x

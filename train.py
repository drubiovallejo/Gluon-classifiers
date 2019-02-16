"""
David Rubio Vallejo
02/27/2019

Main script to run the classifiers form the command line with a number of arguments to be chosen (from hyperparameters
like the learning rate and optimazer, to the batch size, or NN architecture.
"""

# coding: utf-8

import argparse
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon.data import DataLoader

import os
from time import time
from load_data import load_dataset
from CNN_model import CNNTextClassifier
from LSTM_model import LSTMTextClassifier
from RNN_model import RNNTextClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, \
    precision_recall_curve
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data',
                    default=os.path.join(os.getcwd(), 'tweets', 'disaster_tweets_train.tsv'))
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data',
                    default=os.path.join(os.getcwd(), 'tweets', 'disaster_tweets_val.tsv'))
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data',
                    default=os.path.join(os.getcwd(), 'tweets', 'disaster_tweets_test.tsv'))
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.200d',
                    help='Pre-trained embedding source name')

parser.add_argument('--threshold', type=float, default=0.5, help='A float between 0 and 1. The threshold that '
                                                                 'determines the boundary for classification of the '
                                                                 'sigmoid activation function. Any value below it will '
                                                                 'be mapped to pred_label 0, and any value above it '
                                                                 'will be mapped to 1.')
parser.add_argument('--net', type=str, default='cnn', help='Kind of NN: cnn, rnn, lstm.')

args = parser.parse_args()

# Since I'm only outputting one value, use Sigmoid
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()


# In case, the final output is a vector of more than one value
# loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


def train_classifier(vocabulary, data_train, data_val, data_test, ctx):
    """
    Trains the classifier in minibatch fashion with the desired parameters. It also prints statistics on the train,
    val, and test sets.
    """

    # Set up the data loaders for each data source into minibatches of the desired size
    train_dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    # Collects the dimensions of the word vectors so that we can create the NN
    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape

    # determine the kind of NN architecture used
    if args.net == 'cnn':
        model = CNNTextClassifier(emb_input_dim, emb_output_dim)
        print('CNN architecture created...')
    elif args.net == 'lstm':
        model = LSTMTextClassifier(emb_input_dim, emb_output_dim)
        print('LSTM architecture created...')
    elif args.net == 'rnn':
        model = RNNTextClassifier(emb_input_dim, emb_output_dim)
        print('RNN architecture created...')

    # Initialize model parameters on the context ctx
    model.initialize(ctx=ctx)

    # Set the embedding layer parameters to the pre-trained embedding in the vocabulary
    model.embedding.weight.set_data(vocabulary.embedding.idx_to_vec)
    print('Embedding layer populated with word-vectors...')

    # OPTIONAL for efficiency - perhaps easier to comment this out during debugging
    # model.hybridize()

    # this prevents the embedding layer from getting updated through backpropagation, which means that the
    # word-embeddings are not dynamically adapting to the present classification task
    # model.embedding.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(model.collect_params(), optimizer=args.optimizer,
                            optimizer_params={'learning_rate': args.lr})

    print('Begin training...')

    # Variables to keep track of the best values to plot the precision-recall curve of the best model trained
    # (based on who got the best average precision)
    prev_avg_precision = 0
    relative_recall = None
    relative_precision = None

    # store time reference to calculate training time
    t0 = time()

    epoch_num = 1
    for epoch in range(args.epochs):
        train_loss = 0.
        for batch_idx, (data, label) in enumerate(train_dataloader):
            # data is a matrix where each row is a document and the total number of rows is the batch size
            data = data.as_in_context(ctx)
            # label is a list with the labels for each doc in the minibatch
            label = label.as_in_context(ctx)

            with autograd.record():
                output = model(data)
                # Reshapes the output so that it can be compared to the shape of the label properly. I also had to
                # explicitly state that I wanted float64 (rather than the default float32) otherwise I got an error
                loss = loss_fn(output.reshape(label.shape).astype('float64'), label).mean()
            loss.backward()

            trainer.step(args.batch_size)
            train_loss += loss

        # Collects stats for each dataset after each training epoch
        # train_accuracy, train_prec, train_rec, train_f1, train_avg_prec, _ = evaluate(model, train_dataloader)
        # val_accuracy, val_prec, val_rec, val_f1, val_avg_prec, val_loss = evaluate(model, val_dataloader)
        test_accuracy, test_prec, test_rec, test_f1, test_avg_prec, test_loss, test_rel_prec, test_rel_recall = \
            evaluate(model, test_dataloader)

        print()
        print('Epoch {:d}'.format(epoch_num))
        # print('Train loss: {:.2f}; accuracy: {:.2f}; precision {:.2f}; recall {:.2f}; F1 {:.2f}; Avg prec {:.2f}'
        #       .format(train_loss.asscalar(), train_accuracy, train_prec, train_rec, train_f1, train_avg_prec))
        # print('Val loss: {:.2f}; accuracy {:.2f}; precision {:.2f}; recall {:.2f}; F1 {:.2f}; Avg prec {:.2f}'
        #       .format(val_loss, val_accuracy, val_prec, val_rec, val_f1, val_avg_prec))
        print('Test loss: {:.2f}, accuracy {:.2f}; precision {:.2f}; recall {:.2f}; F1 {:.2f}; Avg prec {:.2f}'
              .format(test_loss, test_accuracy, test_prec, test_rec, test_f1, test_avg_prec))

        epoch_num += 1

        # Stores the lists of recall and precision values for creating the precision-recall curve of the BEST epoch
        # I chose as the best, the first model with the highest average precision
        if prev_avg_precision < test_avg_prec:
            prev_avg_precision = test_avg_prec
            relative_recall = test_rel_recall
            relative_precision = test_rel_prec

    t1 = time()

    # Plots the precision recall curve for the model with the best average precision
    plot_precision_recall_curve(relative_precision, relative_recall)

    print()
    print('Total training time {:.2f} seconds'.format(t1 - t0))


def evaluate(model, dataloader, ctx=mx.cpu()):
    """
    Evaluates the predictions on the dataloader items from model.
    Returns accuracy, precision, recall, and F1
    """
    prec = 0
    rec = 0
    acc = 0
    batches = 0
    loss = 0

    # Stores the list of labels and output values for creating the precision-recall curve
    y_trues = []
    y_scores = []

    for batch_idx, (data, label) in enumerate(dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = model(data)

        loss += loss_fn(output.reshape(label.shape).astype('float64'), label).mean()

        # Applies ceiling to output values and threshold. E.g. if threshold is 0.5 and output is 0.4, the deduction will
        # be negative, so ceiling will round up to predicted class 0. Need to apply final sigmoid activation to the
        # output, since I hadn't applied it yet.
        pred_classes = mx.nd.ceil(output.sigmoid() - args.threshold)

        # Collects the true labels and predicted values as they come out
        y_trues.extend(label.asnumpy())
        y_scores.extend(output.sigmoid().squeeze().asnumpy())

        # Sklearn functions wanted the data to be in numpy array format, instead of ndarray
        prec += precision_score(label.asnumpy(), pred_classes.reshape(label.shape).astype('float64').asnumpy())
        rec += recall_score(label.asnumpy(), pred_classes.reshape(label.shape).astype('float64').asnumpy())
        acc += accuracy_score(label.asnumpy(), pred_classes.reshape(label.shape).astype('float64').asnumpy())
        batches += 1

    prec = prec / batches
    rec = rec / batches
    acc = acc / batches
    f1 = 2 * (prec * rec) / (prec + rec)

    relative_precision, relative_recall, threshold = precision_recall_curve(y_trues, y_scores)
    avg_prec = average_precision_score(y_trues, y_scores)

    return acc, prec, rec, f1, avg_prec, loss.asscalar(), relative_precision, relative_recall


def plot_precision_recall_curve(precision, recall):
    """Plots the precision and recall curve with the list of values passsed"""

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.show()


if __name__ == '__main__':

    # load the vocab and datasets (train, val, test)
    vocab, train_dataset, val_dataset, test_dataset = load_dataset(train_file=args.train_file, val_file=args.val_file,
                                                                   test_file=args.test_file,
                                                                   embeddings=args.embedding_source)

    # Embedding options from Glove:
    # print(nlp.embedding.list_sources('glove'))
    # ['glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d',
    # 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d']

    ctx = mx.cpu()  # or mx.gpu(N) if GPU device N is available

    print('Begin creating NN...')
    train_classifier(vocabulary=vocab, data_train=train_dataset, data_val=val_dataset,
                     data_test=test_dataset, ctx=ctx)

"""
David Rubio Vallejo

This script contains the methods to process the input tweets in TSV format. It creates a Gluon vocabulary object with
the word-vectors of the selected kind. Each dataset (train, val, test) is a list of tuples where the first element is
the vector representation of the twit (as a vector of the indices in the vocabulary of each constituent word) and the
second element is either a 1 (relevant) or 0 (irrelevant)
"""

# coding: utf-8

import gluonnlp as nlp
import pandas as pd
from mxnet import nd
from nltk import word_tokenize


def load_dataset(train_file, val_file, test_file, embeddings, max_length=64):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_df = pd.read_csv(train_file, sep='\t', names=['index', 'label', 'text'])
    val_df = pd.read_csv(val_file, sep='\t', names=['index', 'label', 'text'])
    test_df = pd.read_csv(test_file, sep='\t', names=['index', 'label', 'text'])
    print('Datasets loaded...')

    # If we add val and test sentences to the vocabulary, it'll be a Transductive approach (the results in val and test
    # should be better that way)
    vocabulary = build_vocabulary(embeddings, train_df, val_df, test_df)
    # Uncomment for non-transductive approach:
    # vocabulary = build_vocabulary(embeddings, train_df)
    # print('Vocabulary created...')

    train_dataset = preprocess_dataset(train_df, vocabulary, max_length)
    print('Training dataset created...')
    val_dataset = preprocess_dataset(val_df, vocabulary, max_length)
    print('Validation dataset created...')
    test_dataset = preprocess_dataset(test_df, vocabulary, max_length)
    print('Test dataset created...')

    return vocabulary, train_dataset, val_dataset, test_dataset


def build_vocabulary(embeddings, tr_df, val_df=None, tst_df=None):
    """
    Inputs: arrays representing the training, and optionally validation and test data (transductive case)
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []

    # appends the other datasets if they are not null
    datasets = [tr_df]
    if val_df is not None:
        datasets.append(val_df)
    if tst_df is not None:
        datasets.append(tst_df)

    # For each dataset, get each twit, tokenize it, and add each token to the list of tokens
    for dataset in datasets:
        for text_instance in dataset['text'].values:
            tokens = word_tokenize(text_instance)
            all_tokens.extend(tokens)

    # Count the tokens and create a vocab object
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)

    # Attach selected embeddings to the vocabulary
    vocab.set_embedding(nlp.embedding.create('glove', source=embeddings))

    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    label = x.label
    text = x.text

    token_list = text.split()

    # Create a 1D vector of the max_length considered for each twit (here 64 words by default), where each element is
    # the vocab index for that word
    data = nd.zeros((max_len, 1))

    # The 0 entries in 'data' will be replaced by the index of each word. If there are more tokens than 'max_len', we
    # don't take them into account. If there are less, the remaining will remain as 0 vectors (padded)
    for i in range(len(token_list)):
        if i < max_len:
            vec_idx = vocab[token_list[i]]
            data[i] = vec_idx
        else:
            break

    # Translate the true label into a 1 or 0 (of type float for easier comparison in training algorithm)
    if label == 'Relevant':
        label = 1.
    else:
        label = 0.

    # return the transpose of data for easier comparison later on in training
    return data.T, label


def preprocess_dataset(dataset, vocab, max_len):
    """Wrapper method to apply the preprocessing to each twit in the dataset and return a list"""
    preprocessed_dataset = [_preprocess(x, vocab, max_len) for x in dataset.itertuples()]
    return preprocessed_dataset

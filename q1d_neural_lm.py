import os
import random
import time

import numpy as np
import pandas as pd

from data_utils import utils
from sgd import sgd
from q1c_neural import forward, forward_backward_prop

import re


VOCAB_EMBEDDING_PATH = "data/lm/vocab.embeddings.glove.txt"
BATCH_SIZE = 50
NUM_OF_SGD_ITERATIONS = 40000
LEARNING_RATE = 0.3


def load_vocab_embeddings(path=VOCAB_EMBEDDING_PATH):
    result = []
    with open(path) as f:
        index = 0
        for line in f:
            line = line.strip()
            row = line.split()
            data = [float(x) for x in row[1:]]
            assert len(data) == 50
            result.append(data)
            index += 1
    return result


def load_data_as_sentences(path, word_to_num):
    """
    Conv:erts the training data to an array of integer arrays.
      args: 
        path: string pointing to the training data
        word_to_num: A dictionary from string words to integers
      returns
        An array of integer arrays. Each array is a sentence and each 
        integer is a word.
    """
    docs_data = utils.load_dataset(path)
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data

def convert_to_lm_dataset(S):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
    Returns the dataset a bigram prediction problem. For any word, predict the
    next work. 
    IMPORTANT: we have two padding tokens at the beginning but since we are 
    training a bigram model, only one will be used.
    """
    in_word_index, out_word_index = [], []
    for sentence in S:
        for i in range(len(sentence)):
            if i < 2:
                continue
            in_word_index.append(sentence[i - 1])
            out_word_index.append(sentence[i])
    return in_word_index, out_word_index

def shuffle_training_data(in_word_index, out_word_index):
    combined = list(zip(in_word_index, out_word_index))
    random.shuffle(combined)
    return list(zip(*combined))


def int_to_one_hot(number, dim):
    res = np.zeros(dim)
    res[number] = 1.0
    return res


def lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, params):

    data = np.zeros([BATCH_SIZE, input_dim])
    labels = np.zeros([BATCH_SIZE, output_dim])

    # Construct the data batch and run you backpropogation implementation
    ### YOUR CODE HERE
    indices = np.random.choice(len(in_word_index), BATCH_SIZE, replace=False)
    for i, idx in enumerate(indices):
        data[i] = num_to_word_embedding[in_word_index[idx]]  # Get the embedding for input word
        labels[i] = int_to_one_hot(out_word_index[idx], dimensions[2])  # One-hot encode the output word

    cost, grad = forward_backward_prop(data, labels, params, dimensions)
    ### END YOUR CODE

    cost /= BATCH_SIZE
    grad /= BATCH_SIZE
    return cost, grad


def load_data_and_format(eval_data_path, word_to_num, should_preprocess):
    """loads an unformatted corpus to a proper format for the model"""
    docs = []
    # if preprocessing, ignore all non-letter
    # TODO: change the preprocessing it just dont work
    pattern = r"\b\w+(?:-\w+)*\b" if should_preprocess else r"\w+(?:-\w+)*|[^\w\s]"
    with open(eval_data_path, "r", encoding="utf-8") as fd:
        for sentence in fd:
            sentence = sentence[:-1]  # remove \n
            doc = [[token] for token in re.findall(pattern, sentence)]
            docs.append(doc)

    S_data = utils.docs_to_indices(docs, word_to_num)
    return S_data


def eval_neural_lm(eval_data_path, should_format=False, should_preprocess=False):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    should_format: whether to format the file being read (close as possible to
    """
    if should_format:
        S_dev = load_data_and_format(eval_data_path, word_to_num, should_preprocess)
    else:
        _, S_dev = load_data_as_sentences(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    data = np.array(num_to_word_embedding)[in_word_index]
    loss_sum = 0
    for i in range(data.shape[0]):
        loss_sum += np.log(forward(data[i], out_word_index[i], params, dimensions))
    perplexity = (np.e ** -(loss_sum/num_of_examples))[0]  # technically returns a vector of length 1
    ### END YOUR CODE

    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    vocab = pd.read_table("data/lm/vocab.ptb.txt",
                          header=None, sep="\s+", index_col=0, names=['count', 'freq'], )

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
    num_to_word_embedding = load_vocab_embeddings()
    word_to_num = utils.invert_dict(num_to_word)

    # Load the training data
    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_train)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    random.seed(31415)
    np.random.seed(9265)
    in_word_index, out_word_index = shuffle_training_data(in_word_index, out_word_index)
    startTime = time.time()

    # Training should happen here
    # Initialize parameters randomly
    # Construct the params
    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.random.randn((input_dim + 1) * hidden_dim + (
        hidden_dim + 1) * output_dim, )
    print(f"#params: {len(params)}")
    print(f"#train examples: {num_of_examples}")

    # convert to np
    num_to_word_embedding = np.array(num_to_word_embedding)
    in_word_index = np.array(in_word_index)
    out_word_index = np.array(out_word_index)

    # run SGD
    params = sgd(
            lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
            params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print(f"training took {time.time() - startTime} seconds")

    # Evaluate perplexity with dev-data
    perplexity = eval_neural_lm('data/lm/ptb-dev.txt')
    print(f"dev perplexity : {perplexity}")

    # Evaluate perplexity with test-data (only at test time!)
    if os.path.exists('data/lm/ptb-test.txt'):
        process = input("Preprocess test file (y/n)? ").lower() == 'y'
        perplexity = eval_neural_lm('data/lm/ptb-test.txt', True, process)
        print(f"test perplexity : {perplexity}")
    else:
        print("test perplexity will be evaluated only at test time!")

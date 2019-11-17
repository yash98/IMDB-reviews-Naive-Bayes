import numpy as np
import sys
import re
import string

# IMDB specific


def format_train_data(trainfilename, word_freq_thresh):
    trainfile = open(trainfilename, 'r')

    X_list = list()
    Y_list = list()
    trainfile.readline()
    for line in trainfile:
        splitted = line.split('",')
        X_list.append(splitted[0][1:])
        Y_list.append(splitted[-1])

    X_processed = list()
    words = dict()
    words_freq = dict()
    word_id = 0
    # processing
    for x in X_list:
        x_words = list()
        # reg_words = re.sub('['+string.punctuation+']', '', x).split()
        reg_words = re.split(
            r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", x)
        for reg_word in reg_words:
            f = reg_word.lower()
            x_words.append(f)
            if f not in words:
                words[f] = word_id
                words_freq[f] = 1
                word_id += 1
            else:
                words_freq[f] += 1
        X_processed.append(x_words)

    selected_words = dict()

    if type(word_freq_thresh) == float:
        word_freq_thresh *= len(Y_list)
    elif type(word_freq_thresh) == int:
        pass
    else:
        assert "word_freq_thresh must be int or float"

    word_id = 0
    for k in words:
        if words_freq[k] >= word_freq_thresh:
            selected_words[k] = word_id
            word_id += 1

    X = np.zeros((len(Y_list), len(words)))
    Y = np.zeros((len(Y_list),), dtype=int)
    Y_probs = np.zeros((2,))

    count = 0
    for review in X_processed:
        for word in review:
            X[count][words[word]] = 1
        count += 1

    count = 0
    for sentiment in Y_list:
        if sentiment == 'positive':
            Y[count] = 1
            Y_probs[1] += 1
        else:
            Y_probs[0] += 1
        count += 1

    Y_probs /= np.sum(Y_probs)

    trainfile.close()
    return X, Y, Y_probs, selected_words


def train(X, Y, num_classes):
    W = np.zeros((num_classes, X.shape[1]))
    for i in range(Y.shape[0]):
        W[Y[i], :] += X[i, :]
    W /= np.sum(W, axis=0)
    return W


def sentence_to_features(s, words):
    X = np.zeros((1, len(words)))
    reg_words = re.split(
        r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", s)

    for word in reg_words:
        f = word.lower()
        X[words[f]] = 1

    return X


def predict_file(testfilename, writefilename):
    pass


if __name__ == "__main__":
    X, Y, Y_probs, words = format_train_data(sys.argv[1], 0.02)
    # W = train(X, Y, 2)

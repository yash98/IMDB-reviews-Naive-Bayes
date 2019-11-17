import numpy as np
import sys
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()

ignores = {'whom', "you're", 'but', 'at', 'more', 'themselves', 'couldn', 'why', 'few', "you've", 'your',
             'doesn', 'before', 'him', 'she', 'don', 'what', 'are', 'doing', 'theirs', 'all', 've', 'into',
             'himself', 'same', 'has', 'above', 'was', 'where', "wasn't", 'the', 'just', 'again', 'm', 'isn',
             'did', 'does', 'or', "won't", 'yourself', 'here', 'll', 'through', 'because', 'about', 'which',
             'ours', 'herself', 'my', 'this', 'how', 'during', 'until', 'between', 'aren', 'ma', 'be', 'of',
             'some', 'wasn', 'too', "didn't", "isn't", "doesn't", 'then', 'itself', "mightn't", 'd', 'from',
             'them', 'can', 'each', 'no', 'ourselves', 'won', 'ain', 'yours', 'myself', 'such', 'to',
             "needn't", 'mustn', "should've", 'when', 'weren', 'shouldn', 'haven', "mustn't", 'in', 'we',
             'against', 'both', 'needn', 'those', 'an', 'only', "that'll", 'hers', 'with', 'by', 'will',
             'wouldn', 'had', 'while', 'out', "you'll", 'not', 't', "weren't", "hadn't", 'hadn', 'if', 's',
             "hasn't", 'mightn', 'any', 'their', 'were', 'having', 'now', 'hasn', 'it', 'so', 'its', 'he',
             'y', 'should', 'you', 'me', 'yourselves', 'a', 'off', "haven't", "you'd", 'i', 'our', 'who',
             'further', 'is', 'very', 'her', "shouldn't", 'am', 'o', 're', 'over', 'shan', 'once', 'for',
             'been', 'there', 'own', "shan't", 'on', 'down', 'do', "wouldn't", 'his', 'these', 'most', 'that',
             "it's", 'and', 'nor', "don't", 'other', "she's", 'after', 'below', 'didn', "aren't", 'they',
             'being', "couldn't", 'have', 'than', 'up', 'as', 'under'}


def dict_update(d, word):
    if word in d:
        d[word] += 1
    else:
        d[word] = 1


def train_on_file(trainfilename, lower_word_freq_thresh, upper_word_freq_thresh):
    # IMDB specific
    trainfile = open(trainfilename, 'r')

    X_list = list()
    Y_list = list()
    trainfile.readline()
    for line in trainfile:
        splitted = line.split('",')
        X_list.append(splitted[0][1:])
        if splitted[-1] == 'positive\n':
            Y_list.append(1)
        else:
            Y_list.append(0)

    Y = np.array(Y_list, dtype=int)
    words_freq = [dict(), dict()]

    count = 0
    for x in X_list:
        y = Y[count]
        unique = dict()
        # reg_words = re.sub('['+string.punctuation+']', '', x).split()
        reg_words = re.split(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", x)
        for reg_word in reg_words:
            f = reg_word.lower()
            f = stemmer.stem(f)
            if f not in ignores:
                if f not in unique:
                    dict_update(words_freq[y], f)
                    unique[f] = 1
        count += 1

    lower_word_freq_thresh *= len(Y_list)
    upper_word_freq_thresh *= len(Y_list)

    words_freq_processed = [dict(), dict()]
    for i in range(len(words_freq)):
        count_of_words_in_class = 0
        for k in words_freq[i]:
            f = words_freq[i][k]
            if upper_word_freq_thresh > f >= lower_word_freq_thresh:
                count_of_words_in_class += f
                words_freq_processed[i][k] = f
        for k in words_freq_processed[i]:
            # words_freq[i][k] /= count_of_words_in_class
            words_freq_processed[i][k] /= count_of_words_in_class
            # words_freq[i][k] = np.log(words_freq[i][k]/count_of_words_in_class)

    Y_probs = np.zeros((2,))
    for y in Y:
        Y_probs[y] += 1

    Y_probs /= np.sum(Y_probs)
    # Y_probs = np.log(Y_probs/np.sum(Y_probs))

    trainfile.close()
    return X_list, Y, Y_probs, words_freq_processed


def predict(s, word_probs, class_probs):
    reg_words = re.split(r"(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", s)
    unique = dict()
    probs = np.zeros(class_probs.shape)
    probs = class_probs.copy()
    probs += 1.0
    for word in reg_words:
        f = word.lower()
        f = stemmer.stem(f)
        if f not in unique:
            unique[f] = 1
            for i in range(class_probs.shape[0]):
                if f in word_probs[i]:
                    # probs[i] += word_probs[i][word]
                    probs[i] *= word_probs[i][f] + 1.0
    return probs


def predict_file(testfilename, word_probs, class_probs, writefilename):
    testfile = open(testfilename, 'r')
    testfile.readline()
    predictions = []
    for line in testfile:
        predictions.append(
            np.argmax(predict(line[1:-1], word_probs, class_probs)))
    np.savetxt(writefilename, np.array(predictions), fmt='%i')


if __name__ == "__main__":
    X_list, Y, class_probs, word_probs = train_on_file(sys.argv[1], 0.05, 1.0)
    print(word_probs)
    print(len(word_probs[0]), len(word_probs[1]))
    predict_file(sys.argv[2], word_probs, class_probs, sys.argv[3])

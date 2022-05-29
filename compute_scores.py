import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def compute_sent_score(args):
    with open(args.sent_output) as fout_sent:
        sent_outputs = pd.Series([float(line.strip()) for line in fout_sent])
    with open(args.test_hter, 'r', encoding='utf-8') as fhter:
        hter = pd.Series([float(x.strip()) for x in fhter])
    pearson = sent_outputs.corr(hter)
    print(f'Pearson: {pearson:.6f}')


def compute_word_score(args):
    with open(args.word_output) as fout_word:
        word_outputs = [[int(word == 'OK') for word in line.strip().split(' ')] for line in fout_word]
    with open(args.test_tags) as flabels:
        labels = [x.strip().split(' ')[1::2] for x in flabels]

    # Flatten
    y_pred = np.array([word for sent in word_outputs for word in sent])
    y_true = np.array([int(word == 'OK') for sent in labels for word in sent])

    f1_ok = f1_score(y_true, y_pred)
    f1_bad = f1_score(y_true, y_pred, pos_label=0)

    f1_mult = f1_ok * f1_bad
    print(f'F1_OK: {f1_ok:.6f} F1_Bad: {f1_bad} F1_Mult: {f1_mult}')


def main(args):
    compute_sent_score(args)
    compute_word_score(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-hter', type=str, default='data 20qe/test/test.hter')
    parser.add_argument('--test-tags', type=str, default='data 20qe/test/test.tags')

    parser.add_argument('--sent-output', type=str, default='models/20qe/test/sent_output.txt')
    parser.add_argument('--word-output', type=str, default='models/20qe/test/word_output.txt')

    args = parser.parse_args()
    #print(args)

    main(args)
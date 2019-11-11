from __future__ import division
from codecs import open

def read_document(document):
    docs = []
    labels = []
    with open(document, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
        return docs, labels

all_docs, all_labels = read_document('all_Sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]


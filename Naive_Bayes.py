from __future__ import division
from codecs import open
from collections import Counter
import math

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

def train_documents(documents,label):
    dict_log_prob ={}
    dict_total_words =Counter()
    for i in range(len(documents)):
        if label[i] not in dict_log_prob:
            dict_log_prob[label[i]] = Counter()
        for members in documents[i]:
            for labels in dict_log_prob:
                dict_log_prob[labels][members] = 0.5
    for j in range(len(documents)):
        for members in documents[j]:
            dict_log_prob[label[j]][members] += 1
            dict_total_words[label[j]]+=1
    for label in dict_log_prob:
        dict_total_words[label]=dict_total_words[label]+0.5*len(dict_log_prob[label])
    for dict_label in dict_log_prob:
        for word in dict_log_prob[dict_label]:
            dict_log_prob[dict_label][word] = dict_log_prob[dict_label][word] /math.log(dict_total_words[dict_label])
    total=0
    return dict_log_prob

print(train_documents(all_docs,all_labels))

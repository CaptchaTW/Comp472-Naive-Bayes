from __future__ import division
from codecs import open
from collections import Counter
import math
from sys import maxsize
def read_document(document):
    docs = []
    labels = []
    with open(document, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
        return docs, labels



def train_documents(documents,label):
    dict_log_prob ={}
    dict_label_log_prob =Counter()
    dict_total_words =Counter()
    total_label = 0
    for i in range(len(documents)):
        dict_label_log_prob[label[i]] +=1
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
            dict_log_prob[dict_label][word] = math.log(dict_log_prob[dict_label][word] / dict_total_words[dict_label])

    for dict_label in dict_label_log_prob:
        total_label=   total_label+ dict_label_log_prob[dict_label]
    for dict_label in dict_label_log_prob:
        dict_label_log_prob[dict_label] = math.log(dict_label_log_prob[dict_label]/total_label)
    return dict_log_prob, dict_label_log_prob

def score_doc_label(document,label,log_prob,label_log_prob):
    total_log_prob =0
    for words in document:
        total_log_prob +=log_prob[label][words]
    total_log_prob+=label_log_prob[label]
    return total_log_prob

def classify_nb(document,log_prob,label_log_prob):
    label = None
    score = -maxsize
    for labels in label_log_prob:
        log_score = score_doc_label(document,labels,log_prob,label_log_prob)
        if score < log_score:
            score = log_score
            label = labels
    return label

def classify_documents(docs,log_prob,label_log_prob):
    dict_label = []
    for document in docs:
        dict_label.append(classify_nb(document,log_prob,label_log_prob))
    return dict_label

def accuracy(true_labels,guessed_labels):
    correct_guess = 0
    total_guess = 0
    list_index = []
    for i in range(len(true_labels)):
        total_guess +=1
        if true_labels[i] == guessed_labels[i]:
            correct_guess+=1
        else:
            list_index.append(i)
    return correct_guess/total_guess,list_index

def class_accuracy(true_labels,guessed_labels):
    class_list = {}
    for members in true_labels:
        if members not in class_list:
            class_list[members] = 1
        else:
            class_list[members]+=1
    for members in class_list:
        correct_guess = 0
        for i in range(len(true_labels)):
            if true_labels[i] == members:
                if true_labels[i] == guessed_labels[i]:
                    correct_guess += 1
        class_list[members] = correct_guess/class_list[members]
    return class_list
def find_high_error(documents,index,log_prob,log_label_prob):
    list_wrong_score = []
    extreme_list =[]
    indexing = None
    counter = 0
    while counter!=4:
        score1 = maxsize

        for indexes in index:
            if indexes not in extreme_list:
                score = 0
                for labels in log_label_prob:
                    score+=score_doc_label(documents[indexes],labels,log_prob,label_log_prob)
                if score<score1:
                    score1 = score
                    indexing = indexes
        extreme_list.append(indexing)
        counter+=1
    return extreme_list
all_docs, all_labels = read_document('all_Sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]
log_prob,label_log_prob = train_documents(train_docs,train_labels)
guessed_labels = classify_documents(eval_docs,log_prob,label_log_prob)
accuracy,index = accuracy(eval_labels,guessed_labels)
list_high_innacuracy = find_high_error(eval_docs,index,log_prob,label_log_prob)
class_list_accuracy = class_accuracy(eval_labels,guessed_labels)
print("The accuracy of the classifier on the test set is: ", end = '')
print(accuracy)
for item in list_high_innacuracy:
    print("The correct label is:", end = '')
    print(eval_labels[item])
    print("The guessed label is:", end = '')
    print(guessed_labels[item])
    print("The guessed document is:", end = '')
    print(eval_docs[item])
#Task 4:
#
print("The accuracy of each class is:",end = '')
print(class_list_accuracy)

import itertools

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from argparse import Namespace
import matplotlib.pyplot as plt
import os
import pickle

args = Namespace(
    # file_path
    train_file='data/full_data/full_train_data.csv',
    test_file='data/full_data/full_test_data.csv',
    model_path='models/svm/',
    model_file='svm_reuters_v{}.sav',
    version=0.1,
    # training
    remove_stopword=True,
    # option
    seed=1337,
    expand_file=True
)

if args.expand_file:
    args.model_file = os.path.join(args.model_path, args.model_file.format(args.version))

english_stopword = list(stopwords.words('english'))
np.random.seed(args.seed)

# extract feature tf-idf
train_df = pd.read_csv()
train_text = train_df.text.to_list()
train_target = train_df.category.to_list()

if args.remove_stopword:
    tfidf = TfidfVectorizer(lowercase=True, stop_words=english_stopword)
else:
    tfidf = TfidfVectorizer(lowercase=True)
tfidf.fit(train_text)
train_tfidf = tfidf.transform(train_text)

# encode label
encoder = LabelEncoder()
encoder.fit(train_target)
y_train = encoder.transform(train_target)

# training with LinearSVC()
''' use SVM for classification'''
SVM = svm.LinearSVC()
SVM.fit(train_tfidf, y_train)

# save model
pickle.dump(SVM, open(args.model_file, 'wb'))

del train_tfidf

test_df = pd.read_csv('data/full_data/full_test_data.csv')
test_text = train_df.text.to_list()
test_target = train_df.category.to_list()
y_test = LabelEncoder().fit_transform(test_target)
test_tfidf = tfidf.transform(test_text)

# evaluate model
# Accuracy
y_pred = SVM.predict(test_tfidf)
print(SVM)
print("SVM Accuracy Score -> ", accuracy_score(y_pred, y_test)*100)

# Confusion matrix
classes = encoder.classes_
# print(classes)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(
        accuracy, misclass))
#     plt.show()
    if not os.path.exists("results/"):
        os.mkdir("results")
    if normalize:
        plt.savefig('results/confusion_matrix_normalize.png')
    else:
        plt.savefig('results/confusion_matrix.png')
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
plot_confusion_matrix(cm, normalize=True, target_names=classes, title="Confusion Matrix Reuters10 Classification(Normalize)")
plot_confusion_matrix(cm, normalize=False, target_names=classes, title="Confusion Matrix Reuters10 Classification")

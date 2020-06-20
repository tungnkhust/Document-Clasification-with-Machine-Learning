import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from argparse import Namespace
import os
import pickle
from utils import *
from preprocessing import *

class Classifier(object):
    def __init__(self, args, processor: TextProcessor=None):
        self.args = args
        self.train_df = pd.read_csv(args.train_file)
        self.texts = self.train_df.text.tolist()
        self.labels = self.train_df.label.tolist()
        self.labelencoder = LabelEncoder().fit(sorted(self.train_df.label.tolist()))

        vocab = None
        stopwords = None

        if processor is None:
            processor = TextProcessor()
        self.processor = processor

        if args.stopword_file != '':
            stopwords = self.load_stopword(args.stopword_file)
            
        if args.vocab_file != '':
            vocab = self.load_vocab(args.vocab_file)
        self.vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words=stopwords)

        if args.kernel == 'linear':
            self.classifier = svm.LinearSVC(C=args.C, random_state=42)
        else:
            self.classifier = svm.SVC(C=args.C, kernel=args.kernel, gamma=args.gamma, random_state=42)

    def train(self):
        self.vectorizer.fit(self.train_df.text.tolist())
        X = self.vectorizer.transform(self.texts)
        self.classifier.fit(X, self.labelencoder.transform(self.labels))

    def predict(self, text):
        text = self.processor.transform(text)
        tfidf = self.vectorizer.transform([text])
        return self.classifier.predict([text])

    def predict_all(self, raw_documents):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.predict(tfidf)

    def sorce(self, raw_documents, y_targets):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.score(tfidf, y_targets)

    def load_vocab(self, vocab_file):
        vocab_df = pd.read_csv(vocab_file)
        vocab = vocab_df.vocab.tolist()
        return {word: index for index, word in enumerate(vocab)}

    def load_stopword(self, stopword_file):
        with open(stopword_file, 'r') as pf:
            stopwords = pf.readlines()
            stopwords = [word.replace('\n', '') for word in stopwords]
        return stopwords

    def evaluate(self, test_df_or_testfile):
        test_df = test_df_or_testfile
        if os.path.exists(test_df_or_testfile):
            test_df = pd.read_csv(test_df_or_testfile)
        texts = test_df.text.to_list()
        labels = test_df.label.to_list()
        y_labels = self.labelencoder.transform(labels)
        y_pred = self.predict_all(texts)
        print(self.classifier)
        print("Accuracy Score: ", accuracy_score(y_true=y_labels, y_pred=y_pred) * 100)

        classes = self.labelencoder.classes_
        pred_labels = self.labelencoder.inverse_transform(y_pred)
        cm = confusion_matrix(y_true=labels, y_pred=pred_labels, labels=classes)

        plot_confusion_matrix(cm, normalize=False, target_names=classes,
                              title="Confusion Matrix")

        plot_confusion_matrix(cm, normalize=True, target_names=classes,
                              title="Confusion Matrix(Normalize)")


def main():
    args = Namespace(
        # file_path
        train_file='data/full_data/data.csv',
        test_file='data/full_data/test.csv',
        model_path='model_storage/svm/',
        model_file='svm_{}_v{}.pkl',
        version=0.1,

        # feature extacter
        vocab_file='',
        stopword_file='',

        # SVM
        kernel='linear',
        C=1,
        gammar=100,

        # option
        seed=1337,
        expand_file=True
    )

    clf = Classifier(args)
    clf.train()
    clf.evaluate(args.test_file)

if __name__ == '__main__':
    main()



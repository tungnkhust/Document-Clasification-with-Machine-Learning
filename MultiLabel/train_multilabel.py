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
    def __init__(self, args, train_df, processor: TextProcessor = None, model = 'all', classifier=None, labelencoder=None):
        self.args = args
        self.train_df = train_df
        self.texts = self.train_df.text.tolist()
        self.labels = self.train_df.label.tolist()

        if labelencoder is None:
            labelencoder = LabelEncoder().fit(sorted(set(self.labels)))
        self.labelencoder = labelencoder

        vocab = None
        stopwords = None

        if processor is None:
            processor = TextProcessor()
        self.processor = processor

        if args.stopword_file != '':
            stopwords = self.load_stopword(args.stopword_file)

        if args.vocab_file != '':
            vocab = self.load_vocab(args.vocab_file)
        self.vectorizer = TfidfVectorizer(
            vocabulary=vocab, stop_words=stopwords, ngram_range=args.ngram_range, sublinear_tf=args.sublinear_tf)

        if classifier is None:
            if model == 'food':
                # classifier = DecisionTreeClassifier(max_depth=3)
                classifier = svm.LinearSVC(C=args.C, random_state=42)
            elif model == 'boat':
                # classifier = svm.LinearSVC(C=1, class_weight={0: 0.05, 1: 0.2, 2: 0.35}, max_iter=1000000)
                # class_weight = {
                #     5: 0.05,
                #     6: 0.2,
                #     16: 0.35
                # }
                # classifier = svm.LinearSVC(C=1,class_weight = class_weight, max_iter=1000000)
                # classifier = svm.LinearSVC(C=1, max_iter=1000000)
                classifier = svm.LinearSVC(C=args.C, random_state=42)
            elif args.kernel == 'linear':
                classifier = svm.LinearSVC(C=args.C, random_state=42)
            else:
                classifier = svm.SVC(C=args.C, kernel=args.kernel, random_state=42)
        self.classifier = classifier

    @classmethod
    def from_csv(cls, args, csv_path, processor: TextProcessor = None, classifier=None):
        data_df = pd.read_csv(csv_path)

        data_df = data_df.sort_values(['text', 'label'])
        data_df = data_df.drop_duplicates()
        textlast = ''
        id = -1
        for index in data_df.index:
            textnow = data_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                data_df['label'][id] += data_df['label'][index]
                data_df.drop(index=index, inplace=True)

        for df in list(set(data_df.label.tolist())):
            if len(data_df[data_df.label == df]) <= 20:
                data_df = data_df[data_df.label != df]

        return cls(args, data_df, processor, classifier)

    def train(self):
        self.vectorizer.fit(self.train_df.text.tolist())
        X = self.vectorizer.transform(self.texts)
        print(X.shape)
        self.classifier.fit(X, self.labelencoder.transform(self.labels))

    def predict(self, text):
        text = self.processor.transform(text)
        tfidf = self.vectorizer.transform([text])
        return self.classifier.predict([text])

    def predict_all(self, raw_documents):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.predict(tfidf)

    def predict_label(self, raw_documents):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        preds = self.classifier.predict(tfidf)
        return self.labelencoder.inverse_transform(preds)

    def sorce(self, raw_documents, y_targets):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.score(tfidf, y_targets)

    def load_vocab(self, vocab_file):
        try:
            vocab_df = pd.read_csv(vocab_file)
            vocab = vocab_df.vocab.tolist()
            return {word: index for index, word in enumerate(vocab)}
        except FileNotFoundError:
            print('Vocab file is not exist!')
            return None

    def load_stopword(self, stopword_file):
        try:
            with open(stopword_file, 'r') as pf:
                stopwords = pf.readlines()
                stopwords = [word.replace('\n', '') for word in stopwords]
            return stopwords
        except FileNotFoundError:
            print('Stopword file is not exist!')
            return None

    def evaluate(self, test_df_or_testfile, show_cm_matrix=False):
        test_df = None
        if isinstance(test_df_or_testfile, str):
            if os.path.exists(test_df_or_testfile):
                test_df = pd.read_csv(test_df_or_testfile)
        else:
            test_df = test_df_or_testfile
        test_df = test_df.sort_values(['text', 'label'])

        test_df = test_df.drop_duplicates()

        textlast = ''
        id = -1
        for index in test_df.index:
            textnow = test_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                test_df['label'][id] += test_df['label'][index]
                test_df.drop(index=index, inplace=True)

        a = len(test_df.label.tolist())
        train_labels = list(sorted(set(self.train_df.label.to_list())))
        labels = test_df.label.to_list()
        for lb in list(set(labels)):
            if train_labels.count(lb) == 0:
                test_df = test_df[test_df.label != lb]

        texts = test_df.text.to_list()
        labels = test_df.label.to_list()

        b = len(test_df.label.tolist())
        y_labels = self.labelencoder.transform(labels)
        y_pred = self.predict_all(texts)
        score(y_true=y_labels, y_pred=y_pred)

        classes = self.labelencoder.classes_
        pred_labels = self.labelencoder.inverse_transform(y_pred)
        cm = confusion_matrix(
            y_true=labels, y_pred=pred_labels, labels=classes)

        plot_confusion_matrix(cm, normalize=False, target_names=classes,
                              title="Confusion Matrix", show=show_cm_matrix)
        plt.show()

        plot_confusion_matrix(cm, normalize=True, target_names=classes,
                              title="Confusion Matrix(Normalize)", show=show_cm_matrix)
        plt.show()


class HierClassifier():
    def __init__(self, args, data_df, modelALL=None, modelFood=None, modleBoat=None, modelMoney=None):
        food_df, boat_df, money_df = self.split_data(data_df)
        # all_df, food_df, boat_df, money_df = self.load_data()
        labelencoder = LabelEncoder().fit(sorted(set(data_df.label.tolist(
        )+food_df.label.tolist()+boat_df.label.tolist()+money_df.label.tolist())))
        self.labelenoder = labelencoder
        self.args = args
        if modelALL is None:
            self.clfAll = Classifier(args, data_df)
        else:
            self.clfAll = Classifier(args, data_df, classifier=modelALL)

        if modelFood is None:
            self.clfFood = Classifier(args, food_df, labelencoder=labelencoder, model = 'food')
        else:
            self.clfFood = Classifier(
                args, food_df, classifier=modelFood, labelencoder=labelencoder, model = 'food')
        if modleBoat is None:
            self.clfBoat = Classifier(args, boat_df, labelencoder=labelencoder, model = 'boat')
        else:
            self.clfBoat = Classifier(
                args, boat_df, classifier=modleBoat, labelencoder=labelencoder, model = 'boat')

        if modelMoney is None:
            self.clfMoney = Classifier(
                args, money_df, labelencoder=labelencoder)
        else:
            self.clfMoney = Classifier(
                args, money_df, classifier=modelMoney, labelencoder=labelencoder)

    def split_data(self, data_df):
        all_df = data_df.copy()
        textlast = ''
        id = -1
        all_df = all_df.sort_values(['text', 'label'])
        all_df = all_df.drop_duplicates()
        for index in all_df.index:
            textnow = all_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                all_df['label'][id] += all_df['label'][index]
                all_df.drop(index=index, inplace=True)

        for df in list(set(all_df.label.tolist())):
            if len(all_df[all_df.label == df]) <= 20:
                all_df = all_df[all_df.label != df]

        food_df = all_df.loc[(all_df['label'] == 'grainwheat') | (
            all_df['label'] == 'corngrain') | (all_df['label'] == 'grain') | (all_df['label'] == 'corngrainwheat')]

        money_df = all_df.loc[(all_df['label'] == 'interest') | (
            all_df['label'] == 'money-fx') | (all_df['label'] == 'trade')| (all_df['label'] == 'interestmoney-fx')| (all_df['label'] == 'money-fxtrade')]

        boat_df = all_df.loc[(all_df['label'] == 'crude')
                              | (all_df['label'] == 'ship') | (all_df['label'] == 'crudeship')]
        
        return food_df, boat_df, money_df

    def load_data(self, all_path=None, food_path=None, boat_path=None, money_path=None):
        if all_path is None:
            all_path = 'data/sub_data/train/all.csv'
        if food_path is None:
            food_path = 'data/sub_data/train/food.csv'
        if boat_path is None:
            boat_path = 'data/sub_data/train/boat.csv'
        if money_path is None:
            money_path = 'data/sub_data/train/money.csv'

        try:
            all_df = pd.read_csv(all_path)
            food_df = pd.read_csv(food_path)
            boat_df = pd.read_csv(boat_path)
            money_df = pd.read_csv(money_path)
            return all_df, food_df, boat_df, money_df
        except FileNotFoundError:
            print('File is not exist!')

    def train(self):
        self.clfAll.train()
        self.clfFood.train()
        self.clfBoat.train()
        self.clfMoney.train()

    def evaluate(self, test_df_or_testfile, show_cm_matrix=False):
        data_df = test_df_or_testfile
        if isinstance(test_df_or_testfile, str):
            if os.path.exists(test_df_or_testfile):
                data_df = pd.read_csv(test_df_or_testfile)

        pred_all = self.clfAll.predict_label(data_df.text.tolist())

        data_df['pred'] = pred_all

        food_df = data_df.loc[(data_df['label'] == 'wheat') | (
            data_df['label'] == 'corn') | (data_df['label'] == 'grain')]
        food_df = food_df.sort_values(['text', 'label'])
        food_df = food_df.drop_duplicates()
        textlast = ''
        id = -1
        for index in food_df.index:
            textnow = food_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                food_df['label'][id] += food_df['label'][index]
                food_df.drop(index=index, inplace=True)

        money_df = data_df.loc[(data_df['label'] == 'interest') | (
            data_df['label'] == 'money-fx') | (data_df['label'] == 'trade')]
        money_df = money_df.sort_values(['text', 'label'])
        money_df = money_df.drop_duplicates()
        textlast = ''
        id = -1
        for index in money_df.index:
            textnow = money_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                money_df['label'][id] += money_df['label'][index]
                money_df.drop(index=index, inplace=True)

        boat_df = data_df.loc[(data_df['label'] == 'crude')
                              | (data_df['label'] == 'ship')]
        boat_df = boat_df.sort_values(['text', 'label'])
        boat_df = boat_df.drop_duplicates()
        textlast = ''
        id = -1
        for index in boat_df.index:
            textnow = boat_df['text'][index]
            if textlast != textnow:
                textlast = textnow
                id = index
            else:
                boat_df['label'][id] += boat_df['label'][index]
                boat_df.drop(index=index, inplace=True)

        data_df = data_df.loc[(data_df['label'] == 'earn')
                              | (data_df['label'] == 'acq')]

        # data_df = data_df[data_df.pred != 'boat']
        # data_df = data_df[data_df.pred != 'money']

        pred_food2 = self.clfFood.predict_label(food_df.text.to_list())
        food_df['pred'] = pred_food2

        pred_boat2 = self.clfBoat.predict_label(boat_df.text.to_list())
        boat_df['pred'] = pred_boat2

        pred_money2 = self.clfMoney.predict_label(money_df.text.to_list())
        money_df['pred'] = pred_money2

        final_df = pd.concat([data_df, food_df, boat_df, money_df])

        preds = final_df.pred.to_list()
        labels = final_df.label.to_list()

        labelencoder = LabelEncoder().fit(labels)
        y_label = labelencoder.transform(labels)
        y_preds = labelencoder.transform(preds)

        score(y_true=y_label, y_pred=y_preds)
        classes = labelencoder.classes_
        cm = confusion_matrix(y_true=labels, y_pred=preds, labels=classes)

        plot_confusion_matrix(cm, normalize=False, target_names=classes,
                              filename='Hire_confusion_matrix.png',
                              title="Confusion Matrix",
                              show=show_cm_matrix)
        plt.show()

        plot_confusion_matrix(cm, normalize=True, target_names=classes,
                              filename='Hire_confusion_matrix.png',
                              title="Confusion Matrix(Normalize)",
                              show=show_cm_matrix)
        plt.show()

    @classmethod
    def from_csv(cls, args, csv_path, modelALL=None, modelFood=None, modleBoat=None, modelMoney=None):
        data_df = pd.read_csv(csv_path)
        return cls(args, data_df, modelALL, modelFood, modleBoat, modelMoney)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default='data/full_data/data_multilabel.csv')
    parser.add_argument('--test_file', type=str,
                        default='data/full_data/test_multilabel.csv')
    parser.add_argument('--vocab_file', type=str, default='')
    parser.add_argument('--stopword_file', type=str, default='')
    parser.add_argument('--ngram_range', type=tuple, default=(1, 1))
    parser.add_argument('--sublinear_tf', type=bool, default=True)
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--C', type=float, default=1)
    parser.add_argument('--hier', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--show_cm_matrix', type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.hier:
        clf = HierClassifier.from_csv(args, args.train_file)
        print('HierClassifier')
    else:
        print('Classifier')
        clf = Classifier.from_csv(args, args.train_file)
    clf.train()
    clf.evaluate(args.test_file, args.show_cm_matrix)

if __name__ == '__main__':
    main()
    print('Done!')

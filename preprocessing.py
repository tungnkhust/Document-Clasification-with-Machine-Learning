import re
import os
import collections
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def built_vocab(data_df, cutoff=25, outfile='vocab/vocab.csv'):
    word_counts = Counter()
    vocab = []
    for text in data_df.text:
        for token in text.split(" "):
            if token not in string.punctuation:
                word_counts[token] += 1

    for word, word_count in word_counts.items():
        if word_count >= cutoff:
            vocab.append(word)
    vocab_df = pd.DataFrame(vocab, columns=['vocab'])
    if os.path.exists('vocab') is False:
        os.mkdir('vocab')
    vocab_df.to_csv(outfile, index=False)
    return vocab


def folder_to_csv(pathdir=None, pathcsv="data/full_data/train.csv"):
    final_list = []
    labels = os.listdir(pathdir)
    for label in labels:
        filenames = os.listdir(pathdir + "/" + label)
        for filename in filenames:
            tmpfile = pathdir + "/" + label + "/" + filename
            with open(tmpfile, 'r') as pf:
                text = pf.read()
            final_list.append({"text": text, "label": label})
    data_df = pd.DataFrame(final_list)
    data_df.to_csv(pathcsv, index=False)

def make_hier_data(data_df_or_data_path, type='train'):
    data_df = data_df_or_data_path
    if isinstance(data_df_or_data_path, str):
        if os.path.exists(data_df_or_data_path):
            data_df = pd.read_csv(data_df_or_data_path)
        else:
            return None
    corn_df = data_df[data_df.label == "corn"]
    grain_df = data_df[data_df.label == "grain"]
    wheat_df = data_df[data_df.label == "wheat"]
    food_df = pd.concat([corn_df, grain_df, wheat_df])

    interest_df = data_df[data_df.label == "interest"]
    moneyfx_df = data_df[data_df.label == "money-fx"]
    money_df = pd.concat([interest_df, moneyfx_df])

    crude_df = data_df[data_df.label == "crude"]
    ship_df = data_df[data_df.label == "ship"]
    boat_df = pd.concat([crude_df, ship_df])

    data_df['label'] = data_df.label.replace('grain', 'food')
    data_df['label'] = data_df.label.replace('corn', 'food')
    data_df['label'] = data_df.label.replace('wheat', 'food')
    data_df['label'] = data_df.label.replace('interest', 'money')
    data_df['label'] = data_df.label.replace('money-fx', 'money')
    data_df['label'] = data_df.label.replace('crude', 'boat')
    data_df['label'] = data_df.label.replace('ship', 'boat')

    if os.path.exists('data/sub_data') is False:
        os.mkdir('data/sub_data')

    if os.path.exists('data/sub_data/' + type) is False:
        os.mkdir('data/sub_data/' + type)

    data_df.to_csv('data/sub_data/' + type + '/all.csv')
    food_df.to_csv('data/sub_data/' + type + '/food.csv')
    money_df.to_csv('data/sub_data/' + type + '/money.csv')
    boat_df.to_csv('data/sub_data/' + type + '/boat.csv')
    return data_df, food_df, money_df, boat_df

def split_by_label(data_df, train_split=0.8):
    by_label = collections.defaultdict(list)
    for _, row in data_df.iterrows():
        by_label[row.label].append(row.to_dict())
    train_list = []
    val_list = []
    for _, item_list in sorted(by_label.items()):
        np.random.shuffle(item_list)
        n_total = len(item_list)
        n_train = int(train_split * n_total)
        train_list.extend(item_list[: n_train])
        val_list.extend(item_list[n_train:])
    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)
    return train_df, val_df

class TextProcessor(object):
    def __init__(self, stopword=None, stemmer=None, lemmatizer=None):
        """
            stopword(list): list common word of language
        """
        self.stopword = stopword

        if stemmer is None:
            stemmer = PorterStemmer()
        self.stemmer = stemmer

        if lemmatizer is None:
            lemmatizer = WordNetLemmatizer()
        self.lemmatizer = lemmatizer


    def remove_url(self, text):
        """ Remove url and link. if replace is not None then link is replaced"""
        url_regex = re.compile(r'\bhttps?://\S+\b')
        text = url_regex.sub(r" ", text)
        return text

    def remove_emoji(self, text):
        """ Remove emoji"""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r" ", text)
        return text

    def remove_irr_char(self, text):
        """ Remove irrelevant character"""
        text = re.sub(r"[^a-zA-Z*\s.]", " ", text)
        text = re.sub("[.]", "", text)
        text = re.sub(r"\n", " ", text)
        return text

    def remove_dupspace(self, text):
        """ Remove duplicate"""
        text = re.sub(r"\n{2,}", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def remove_stopword(self, text):
        """ Remove stopword"""
        if self.stopword is None:
            return text
        words = text.split(' ')
        new_text = []
        for word in words:
            if word not in self.stopword:
                new_text.append(word)
        return ' '.join(new_text)

    def stem(self, text):
        tmp = []
        for word in text.split(' '):
            if word.isalpha():
                word = self.stemmer.stem(word)
                tmp.append(word)
        return ' '.join(tmp)

    def lemmatize(self, text):
        tmp = []
        for word in text.split(' '):
            if word.isalpha():
                word = self.lemmatizer.lemmatize(word)
                tmp.append(word)
        return ' '.join(tmp)

    def nomalize_all(self, text):
        text = self.remove_url(text)
        text = self.remove_emoji(text)
        text = self.remove_irr_char(text)
        text = self.remove_dupspace(text)
        text = self.remove_stopword(text)
        text = self.stem(text)
        text = self.lemmatize(text)
        return text

    def transform(self, text):
        """Process text for predict"""
        text = self.remove_url(text)
        text = self.remove_emoji(text)
        text = self.remove_irr_char(text)
        text = self.remove_dupspace(text)
        return text

class DatasetProcessor(object):
    def __init__(self, data_df, processor: TextProcessor=None, labelencoder=None):
        self.data_df = data_df
        self.train_df = None
        self.val_df = None
        if processor is None:
            processor = TextProcessor()
        self.processor = processor

        if labelencoder is None:
            labelencoder = LabelEncoder().fit(sorted(data_df.label.tolist()))
        self.labelencoder = labelencoder

    def remove_url(self):
        """Remove link url from text in data_df"""
        self.data_df['text'] = self.data_df['text'].apply(lambda x: self.processor.remove_url(x))
        return self

    def remove_irr_char(self):
        """Remove irrelevant characters in data_df"""
        self.data_df['text'] = self.data_df['text'].apply(lambda x: self.processor.remove_irr_char(x))
        return self

    def remove_dupspace(self):
        """ Remove duplicate"""
        self.data_df['text'] = self.data_df['text'].apply(lambda x: self.processor.remove_dupspace(x))
        return self

    def lower(self):
        texts = self.data_df['text'].tolist()
        texts = [text.lower() for text in texts]
        self.data_df['text'] = texts
        return self

    def remove_stop_word(self):
        """Remove stop word in data_df"""
        texts = self.data_df['text'].tolist()
        texts = [self.processor.remove_stopword(text) for text in texts]
        self.data_df['text'] = texts
        return self

    def stem(self):
        """Stem all text in data_df"""
        self.data_df['text'] = self.data_df['text'].apply(lambda x: self.processor.stem(x))
        return self

    def lematize(self):
        """Lemmatize all text in data_df"""
        self.data_df['text'] = self.data_df['text'].apply(lambda x: self.processor.lemmatize(x))
        return self

    def nomalize_all_df(self):
        self.data_df["text"] = self.data_df["text"].apply(lambda x: self.processor.nomalize_all(x))
        return self


    @classmethod
    def from_csv(cls, file_path: str, processor:TextProcessor, delimiter:str=None, header:str='infer'):
        """Create an TextProcessor from the in `cols` of `path/csv_name`"""
        data_df = pd.read_csv(Path(file_path), delimiter=delimiter, header=header)
        return cls(data_df, processor)

    @classmethod
    def from_folder(cls, path: str, processor: TextProcessor):
        """Create an TextProcessor form folder(folder include files that is the text of the label"""
        path = path
        labels = os.listdir(path)
        final_list = []
        for label in labels:
            filenames = os.listdir(path + '/' + label)
            for filename in filenames:
                with open(path + '/' + label + '/' + filename, 'r') as pf:
                    text = pf.read()
                    final_list.append({'text': text, 'label': label})
        data_df = pd.DataFrame(final_list)
        return cls(data_df, processor)

    def to_csv(self, out_path: str):
        self.data_df.to_csv(out_path, index=False)
        if self.train_df is not None:
            self.train_df.to_csv('data/full_data/train.csv', index=False)
        if self.val_df is not None:
            self.val_df.to_csv('data/full_data/val.csv', index=False)

    def get_df(self):
        return self.data_df

    def get_text(self, option='train'):
        return self.data_df.text.tolist()

    def get_label(self):
        return self.data_df.label.tolist()

    def __getitem__(self, index):
        item = self.data_df.iloc[index]
        x_text = item.text
        y_target = self.labelencoder.transform([item.label])
        return x_text, y_target

    def split_by_label(self, train_split=0.8):
        by_label = collections.defaultdict(list)
        for _, row in self.data_df.iterrows():
            by_label[row.label].append(row.to_dict())
        train_list = []
        val_list = []
        np.random.seed(42)
        for _, item_list in sorted(by_label.items()):
            np.random.shuffle(item_list)
            n_total = len(item_list)
            n_train = int(train_split * n_total)
            train_list.extend(item_list[: n_train])
            val_list.extend(item_list[n_train:])
        train_df = pd.DataFrame(train_list)
        val_df = pd.DataFrame(val_list)
        self.train_df = train_df
        self.val_df = val_df
        print('Train size:', len(self.train_df))
        print('Val size: ', len(self.val_df))
        return self

    def get_top_n_unigram(self, n=20):
        corpus = self.data_df.text.tolist()
        vec = CountVectorizer().fit(corpus)
        bags_of_words = vec.transform(corpus)
        sum_words = bags_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word,idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        self.common_words = words_freq[:n]
        return words_freq[:n]

    def get_top_n_bigram(self, n=20):
        corpus = self.data_df.text.tolist()
        vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def get_top_n_trigram(self, n=20):
        corpus = self.data_df.text.tolist()
        vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_distribution_of_label(self):
        if os.path.exists('visual') is False:
            os.mkdir('visual')
        data_df = self.data_df
        label_counts = data_df['label'].value_counts()
        ax = label_counts.plot.bar(rot=45)
        ax.set_title('Label Distribution')
        ax.figure.savefig('visual/label_distribution.png')
        plt.show()

    def plot_distribution_of_top_gram(self, n=20, gram='unigram'):
        common_words = []
        rot = 90
        if os.path.exists('visual') is False:
            os.mkdir('visual')
        if gram == 'unigram':
            common_words = self.get_top_n_unigram(n)
            rot = 90
        elif gram == 'bigram':
            common_words = self.get_top_n_bigram(n)
            rot = 45
        elif gram == 'trigram':
            common_words = self.get_top_n_trigram(n)
            rot = 20
        df1 = pd.DataFrame(common_words, columns=['word', 'count'])
        x = df1.groupby('word').sum()['count'].sort_values(ascending=False)
        ax = x.plot.bar(rot=rot)
        ax.set_title('Top {} words in corpus'.format(n))
        ax.figure.savefig('visual/topnwords.png')
        plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/Reuter10/train')
    parser.add_argument('--test_dir', type=str, default='data/Reuter10/test')
    parser.add_argument('--cutoff', type=int, default=0)
    parser.add_argument('--hier', type=bool, default=False)
    args = parser.parse_args()

    print("Cutoff :", args.cutoff)
    if os.path.exists('data/full_data') is False:
        os.mkdir('data/full_data')
    processor = TextProcessor()
    train_dataset = DatasetProcessor.from_folder(path=args.train_dir, processor=processor)
    train_dataset.remove_url()\
            .remove_irr_char()\
            .lower()\
            .remove_dupspace()\
            .to_csv('data/full_data/data.csv')
    vocab = built_vocab(train_dataset.get_df(), cutoff=args.cutoff)
    # print(len(vocab))

    test_dataset = DatasetProcessor.from_folder(path=args.test_dir, processor=processor)
    test_dataset.remove_url()\
            .remove_irr_char()\
            .lower()\
            .remove_dupspace()\
            .to_csv('data/full_data/test.csv')

    if args.hier:
        make_hier_data('data/full_data/data.csv', type='train')
        make_hier_data('data/full_data/test.csv', type='test')
if __name__ == '__main__':
    main()
    print("Done!")
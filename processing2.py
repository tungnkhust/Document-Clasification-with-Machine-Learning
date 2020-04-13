import re
import os
import collections
import numpy as np
import pandas as pd

from argparse import Namespace

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub("\n", " ", text)
    return text

args = Namespace(
    train_data_dir='data/Reuter10/train',
    test_data_dir='data/Reuter10/test',
    output_train_file='data/full_data/full_train_data.csv',
    output_test_file='data/full_data/full_test_data.csv',
    train_split=0.7,
    val_split=0.3,
    little_rate=0.3,
    seed=1337
)

if os.path.exists('data/full_data') is False:
    os.mkdir('data/full_data/')
if os.path.exists('data/lite_data') is False:
    os.mkdir('data/lite_data/')

print("-----------------train------------------")
by_category = collections.defaultdict(list)

categories = os.listdir(args.train_data_dir)
for category in categories:
    filenames = os.listdir(args.train_data_dir + "/" + category)
    for filename in filenames:
        tmpfile = args.train_data_dir + "/" + category + "/" + filename
        with open(tmpfile, 'r') as pf:
            tmp = pf.read()
        by_category[category].append({"category": category, "text": tmp})
final_list = []
little_list = []
for _, item_list in sorted(by_category.items()):
    np.random.shuffle(item_list)

    n_total = len(item_list)
    n_train = int(args.train_split * n_total)
    n_val = int(args.val_split * n_total)
    n_train_little = int(args.little_rate * n_train)
    n_val_little = int(args.little_rate * n_val)
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train:n_train + n_val]:
        item['split'] = 'val'

    # Add to final list
    final_list.extend(item_list)
    little_list.extend(item_list[:n_train_little])
    little_list.extend(item_list[n_train: n_train + n_val_little])

final_data_train = pd.DataFrame(final_list)
little_data = pd.DataFrame(little_list)
print(final_data_train.category.value_counts())
print(final_data_train.head)

final_data_train.to_csv(args.output_train_file, index=False)
little_data.to_csv("data/lite_data/lite_train_data.csv")
print("-----------------test------------------")
# test
by_category = collections.defaultdict(list)

categories = os.listdir(args.test_data_dir)
for category in categories:
    filenames = os.listdir(args.test_data_dir + "/" + category)
    for filename in filenames:
        tmpfile = args.test_data_dir + "/" + category + "/" + filename
        with open(tmpfile, 'r', encoding='utf8') as pf:
            tmp = pf.read()
        by_category[category].append({"category": category, "text":tmp})
final_list = []
for _, item_list in sorted(by_category.items()):
    for item in item_list:
        item['split'] = 'test'
    final_list.extend(item_list)

final_data_test = pd.DataFrame(final_list)

print(final_data_test.category.value_counts())
print(final_data_test.head())
final_data_test.to_csv(args.output_test_file, index=False)


n_little_test = int(args.little_rate * len(final_data_test))
little_data_test = final_data_test[: n_little_test]
little_data_test.to_csv("data/lite_data/lite_test_data.csv")
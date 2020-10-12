from transformers import MobileBertTokenizer
from transformers.modeling_mobilebert import MobileBertModel

from config import PRETRAINED_MODEL_NAME, DATASET_PATH, MODEL_MAX_LENGTH, PADDING

import pandas as pd
import os
import progressbar
import numpy as np

import tensorflow as tf
import torch

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def setup_progress_bar(title='', maxval=0):
    if len(title) > 0:
        print(title)

    bar = progressbar.ProgressBar(
        maxval=maxval,
        widgets=[progressbar.Bar('=', '[', ']'),
                 ' ', progressbar.Percentage()]
    )

    return bar


def get_max_length(column=''):
    max_length = 0

    for filename in os.listdir(DATASET_PATH):
        if filename.endswith('.csv'):
            path = os.path.join(DATASET_PATH, filename)
            df = pd.read_csv(path)

            bar = setup_progress_bar(
                title='Getting max length',
                maxval=len(df[column])
            )
            bar.start()

            for i, sentence in enumerate(df[column]):
                input_ids = mobile_bert_tokenizer.encode(
                    sentence,
                    add_special_tokens=True
                )
                max_length = max(max_length, len(input_ids))
                bar.update(i+1)

            bar.finish()
            break

    return max_length


def preprocessing_data(features=[], labels=[]):
    if len(features) == 0 or len(labels) == 0:
        return None, None, None, None

    headers = features + labels
    text_feature = features[0]
    df = []
    tokenized_input_ids = []
    tokenized_token_type_ids = []
    tokenized_attention_mask = []
    sentiment_labels = []
    input_ids_padding_length = 0
    attention_mask_padding_length = 0
    max_length = get_max_length(column=text_feature)

    for filename in os.listdir(DATASET_PATH):
        if filename.endswith('.csv'):
            path = os.path.join(DATASET_PATH, filename)
            df = pd.read_csv(path)
            df = df[headers]

            bar = setup_progress_bar(
                title='Encoding text tokens',
                maxval=len(df[text_feature])
            )
            bar.start()

            for i, sentence in enumerate(df[text_feature]):
                if i >= 100: break
                encoded_input = mobile_bert_tokenizer.encode_plus(
                    sentence,
                    max_length=max_length,
                    add_special_tokens=True,
                    padding=True,
                    return_attention_mask=True,
                    # return_tensors='tf'
                )

                input_ids = encoded_input['input_ids']
                token_type_ids = encoded_input['token_type_ids']
                attention_mask = encoded_input['attention_mask']

                input_ids_padding_length += max_length - len(input_ids)
                attention_mask_padding_length += max_length - len(attention_mask)

                tokenized_input_ids.append(input_ids)
                tokenized_token_type_ids.append(token_type_ids)
                tokenized_attention_mask.append(attention_mask)

                bar.update(i+1)

            sentiment_labels.append(df[labels])
            bar.finish()
            break

    bar = setup_progress_bar(
        title='Padding input_ids and attention_mask',
        maxval=input_ids_padding_length+attention_mask_padding_length
    )
    padding_progress = 0
    bar.start()

    for input_ids in tokenized_input_ids:
        for i in range(len(input_ids), max_length):
            input_ids.append(PADDING)
            padding_progress += 1
            bar.update(padding_progress)

    for attention_mask in tokenized_attention_mask:
        for i in range(len(attention_mask), max_length):
            attention_mask.append(PADDING)
            padding_progress += 1
            bar.update(padding_progress)

    bar.finish()

    return (
        torch.tensor(np.array(tokenized_input_ids)),
        tokenized_token_type_ids,
        torch.tensor(np.array(tokenized_attention_mask)),
        sentiment_labels,
        df
    )


def get_labels():
    labels = []
    emotions_path = 'data/emotions.txt'
    with open(emotions_path) as f:
        for line in f:
            labels.append(line.replace('\n', ''))
    return labels


def get_features():
    return ['text']


# example_sentence = 'He isn\'t as big, but he\'s still quite popular. I\'ve heard the same thing about his content. Never watched him much.'

# Pretrain MobileBertTokenizer and MobileBertModel
mobile_bert_tokenizer = MobileBertTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME)
mobile_bert_model = MobileBertModel.from_pretrained(PRETRAINED_MODEL_NAME)

# Preprocessing Data
labels_heading = get_labels()
features_heading = get_features()

input_ids, token_type_ids, attention_mask, sentiment_labels, df = preprocessing_data(
    features=features_heading,
    labels=labels_heading
)

# Train MobileBERT model
print('Training MobileBERT model...')
last_hidden_states = mobile_bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

features = last_hidden_states[0][:,0,:].numpy()

print(features)

# X_train, X_test, y_train, y_test = train_test_split()

# clf = SVC(kernel='linear', C=2)
# clf.fit(x_train, x_val)

# # tf.data.Dataset.

# print(x_train, x_val)

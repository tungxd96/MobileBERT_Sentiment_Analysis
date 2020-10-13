from transformers import MobileBertTokenizer
from transformers.modeling_mobilebert import MobileBertModel

import config

import pandas as pd
import os
import progressbar
import numpy as np

import tensorflow as tf
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

    for filename in os.listdir(config.DATASET_PATH):
        if filename.endswith('.csv'):
            path = os.path.join(config.DATASET_PATH, filename)
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
    labeled_scores = []
    input_ids_padding_length = 0
    attention_mask_padding_length = 0
    max_length = get_max_length(column=text_feature)
    memory_size = int(config.MEMORY_1_GB / 10)

    for filename in os.listdir(config.DATASET_PATH):
        if filename.endswith('.csv'):
            path = os.path.join(config.DATASET_PATH, filename)
            df = pd.read_csv(path)
            df = df[headers]

            tokenized_input_ids, tokenized_token_type_ids, tokenized_attention_mask = preprocessing_features(
                df_features=df[text_feature],
                tokenized_input_ids=tokenized_input_ids,
                tokenized_token_type_ids=tokenized_token_type_ids,
                tokenized_attention_mask=tokenized_attention_mask,
                memory_size=memory_size,
                max_length=max_length
            )

            sentiment_labels, labels_score = preprocessing_labels(
                labeled_headings=labels,
                sentiment_labels=sentiment_labels,
                labeled_scores=labeled_scores,
                df=df,
                memory_size=memory_size
            )

            break

    bar = setup_progress_bar(
        title='Padding input_ids and attention_mask...',
        maxval=len(tokenized_input_ids)+len(tokenized_attention_mask)
    )
    padding_progress = 0
    bar.start()

    for input_ids in tokenized_input_ids:
        for i in range(len(input_ids), max_length):
            input_ids.append(config.PADDING)
        padding_progress += 1
        bar.update(padding_progress)

    for attention_mask in tokenized_attention_mask:
        for i in range(len(attention_mask), max_length):
            attention_mask.append(config.PADDING)
        padding_progress += 1
        bar.update(padding_progress)

    bar.finish()

    return (
        torch.tensor(np.array(tokenized_input_ids)),
        tokenized_token_type_ids,
        torch.tensor(np.array(tokenized_attention_mask)),
        np.array(labeled_scores)
    )


def preprocessing_features(
    df_features=[], 
    tokenized_input_ids=[], 
    tokenized_token_type_ids=[], 
    tokenized_attention_mask=[],
    memory_size=100,
    max_length=512
):
    bar = setup_progress_bar(
        title='Preprocessing features',
        maxval=len(df_features)
    )
    bar.start()

    for i, sentence in enumerate(df_features):
        if i >= memory_size:
            break

        encoded_input = mobile_bert_tokenizer.encode_plus(
            sentence,
            max_length=max_length,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True
        )

        input_ids = encoded_input['input_ids']
        token_type_ids = encoded_input['token_type_ids']
        attention_mask = encoded_input['attention_mask']

        tokenized_input_ids.append(input_ids)
        tokenized_token_type_ids.append(token_type_ids)
        tokenized_attention_mask.append(attention_mask)

        bar.update(i+1)

    bar.finish()

    return tokenized_input_ids, tokenized_token_type_ids, tokenized_attention_mask


def preprocessing_labels(
    labeled_headings=[],
    sentiment_labels=[],
    labeled_scores=[], 
    df=[],
    memory_size=100,
):
    bar = setup_progress_bar(
        title='Preprocessing labels...',
        maxval=memory_size * 2
    )
    bar.start()

    for i, classification in enumerate(labeled_headings):
        for j, score in enumerate(df[classification]):
            if j >= memory_size:
                break

            try:
                sentiment_labels[j]
            except:
                sentiment_labels.append([])
                
            sentiment_labels[j].append(score)

    extracted_ones = []
    for i, classifications in enumerate(sentiment_labels):
        for j, score in enumerate(classifications):
            if score == 1:
                if len(extracted_ones) == i:
                    extracted_ones.append([])
                extracted_ones[i].append(j)
            else:
                if j == len(classifications) - 1 and len(extracted_ones) == i:
                    extracted_ones.append([-1])
    
        bar.update(i+1)

    print(extracted_ones)
    bar.finish()

    return sentiment_labels, labeled_scores


def get_labels():
    labels = []
    emotions_path = 'data/emotions.txt'
    with open(emotions_path) as f:
        for line in f:
            labels.append(line.replace('\n', ''))
    return labels


def get_features():
    return ['text']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# example_sentence = 'He isn\'t as big, but he\'s still quite popular. I\'ve heard the same thing about his content. Never watched him much.'

# Pretrain MobileBertTokenizer and MobileBertModel
mobile_bert_tokenizer = MobileBertTokenizer.from_pretrained(
    config.PRETRAINED_MODEL_NAME)

mobile_bert_model = MobileBertModel.from_pretrained(
    config.PRETRAINED_MODEL_NAME)
mobile_bert_model.to(device)

# Preprocessing Data
labels_headings = get_labels()
features_headings = get_features()

input_ids, token_type_ids, attention_mask, labels = preprocessing_data(
    features=features_headings,
    labels=labels_headings
)

input_ids = input_ids.type(torch.LongTensor)
attention_mask = attention_mask.type(torch.LongTensor)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Train MobileBERT model
print('Training MobileBERT model...')
with torch.no_grad():
    last_hidden_states = mobile_bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    features = last_hidden_states[0][:,0,:].cpu().detach().numpy()

    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    # print(X_train, y_train)
    # lr_clf = LogisticRegression()
    # lr_clf.fit(X_train, y_train)
    # score = lr_clf.score(X_test, y_test)
    # print(score)

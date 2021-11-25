
from __future__ import absolute_import

import sys
import os

sys.path.append('./drive/My Drive/CMC/')

#!wandb login

from __future__ import absolute_import

import sys
import os

import shutil

try:
    from dotenv import find_dotenv, load_dotenv
except:
    pass

import argparse

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), './drive/My Drive/CMC/'))
except:
    sys.path.append(os.path.join(os.getcwd(), './drive/My Drive/CMC/'))
    
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
except:
    sys.path.append(os.path.join(os.getcwd(), '../'))
    
import pandas as pd
import numpy as np

import pickle
from collections import Counter
from tqdm import tqdm
import random

import tensorflow as tf
import tensorflow.keras.backend as K
#import tensorflow_addons as tfa

try:
    import wandb
    load_dotenv(find_dotenv())
    wandb.login(key=os.environ['WANDB_API_KEY'])
    from wandb.keras import WandbCallback
    _has_wandb = True
except:
    _has_wandb = False

import tokenizers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig, BertTokenizer

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src import data, models

pd.options.display.max_colwidth = -1

print (_has_wandb)

parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

parser.add_argument('--data_path', type=str, default='./drive/My Drive/CMC/data/', required=False,
                    help='train data')
#parser.add_argument('--data_path', type=str, default='../data/', required=False,
#                    help='train data')
parser.add_argument('--val_data', type=str, default=None, required=False,
                    help='validation data')
parser.add_argument('--test_data', type=str, default=None, required=False,
                    help='test data')

parser.add_argument('--transformer_model_pretrained_path', type=str, default='roberta-base', required=False,
                    help='transformer model pretrained path or huggingface model name')
parser.add_argument('--transformer_config_path', type=str, default='roberta-base', required=False,
                    help='transformer config file path or huggingface model name')
parser.add_argument('--transformer_tokenizer_path', type=str, default='roberta-base', required=False,
                    help='transformer tokenizer file path or huggingface model name')

parser.add_argument('--max_text_len', type=int, default=20, required=False,
                    help='maximum length of text')
parser.add_argument('--max_char_len', type=int, default=100, required=False,
                    help='maximum length of text')
parser.add_argument('--max_word_char_len', type=int, default=20, required=False,
                    help='maximum length of text')

parser.add_argument('--emb_dim', type=int, default=128, required=False,
                    help='maximum length of text')
parser.add_argument('--n_layers', type=int, default=2, required=False,
                    help='maximum length of text')
parser.add_argument('--n_units', type=int, default=128, required=False,
                    help='maximum length of text')

parser.add_argument('--epochs', type=int, default=7, required=False,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=.00003, required=False,
                    help='learning rate')
parser.add_argument('--early_stopping_rounds', type=int, default=4, required=False,
                    help='number of epochs for early stopping')
parser.add_argument('--lr_schedule_round', type=int, default=3, required=False,
                    help='number of epochs for learning rate scheduling')

parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                    help='eval batch size')

parser.add_argument('--model_save_path', type=str, default='./drive/My Drive/CMC/models/model_hindi_ZSL/', required=False,
                    help='seed')

#parser.add_argument('--model_save_path', type=str, default='../models/model_hindi_ZSL/', required=False,
#                    help='seed')

parser.add_argument('--wandb_logging', type=bool, default=True, required=False,
                    help='wandb logging needed')

parser.add_argument('--seed', type=int, default=42, required=False,
                    help='seed')


args, _ = parser.parse_known_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

"""### Prepare data for ZSL on 3 different tasks - Hindi"""

full_data = []
for file in ['IIITH_Codemixed.txt', 'hindi_humour-codemix.txt', 'hindi_sarcasm-codemix.txt']:
    if file == 'IIITH_Codemixed.txt':
        df = pd.read_csv(os.path.join(args.data_path, file), sep='\t',header=None,usecols=[1,2])
        df.columns = ['text','category']
        df['task'] = 'Sentiment'
    elif file == 'hindi_humour-codemix.txt':
        df = pd.read_csv(os.path.join(args.data_path, file), sep='\t',header=None)
        df.columns = ['text','category']
        df['category'] = df['category'].apply(lambda x: "Humor" if x == 1 else "No Humor")
        df['task'] = 'Humor'
    else:
        df = pd.read_csv(os.path.join(args.data_path, file), sep='\t',header=None)
        df.columns = ['text','category']
        df['category'] = df['category'].apply(lambda x: "Sarcasm" if x == 1 else "No Sarcasm")
        df['task'] = 'Sarcasm'
    full_data.append(df)
full_data = pd.concat(full_data, axis=0)
df = full_data.copy()

df['strata'] = df.apply(lambda x: "{}_{}".format(x.category,x.task),axis=1)

df.text = df.text.apply(lambda x: data.preprocessing.clean_tweets(x))

df.category.value_counts()

category_dict = {i:val for i,val in enumerate(df.category.unique().tolist())}
inv_category_dict = {val:i for i,val in category_dict.items()}

category_dict

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
for train_index, test_index in kf.split(df.text, df.strata):
    break

train_df = df.iloc[train_index].reset_index(drop=True)
val_df = df.iloc[test_index].reset_index(drop=True)

kf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
for val_index, test_index in kf2.split(val_df.text, val_df.strata):
    break

test_df = val_df.iloc[test_index].reset_index(drop=True)
val_df = val_df.iloc[val_index].reset_index(drop=True)

print (train_df.shape, val_df.shape, test_df.shape)

zsl_df_train = pd.DataFrame()
j = 0

for i in tqdm(range(train_df.shape[0])):
    zsl_df_train.loc[j,'text'] = train_df.text.iloc[i] #df.category.iloc[i].lower() + " text - " + df.text.iloc[i]
    zsl_df_train.loc[j, "category"] = train_df.category.iloc[i]
    zsl_df_train.loc[j, "target"] = 1
    zsl_df_train.loc[j, "task"] = train_df.task.iloc[i]
    zsl_df_train.loc[j, "actual category"] = train_df.category.iloc[i]
    j += 1
    random_label = random.sample(list(set(train_df[train_df.task == train_df.task.iloc[i]].category.unique().tolist()) - set([train_df.category.iloc[i]])),1)
    zsl_df_train.loc[j,'text'] = train_df.text.iloc[i] #random_label[0].lower() + " text - " + df.text.iloc[i]
    zsl_df_train.loc[j, "category"] = random_label #df.category.iloc[i]
    zsl_df_train.loc[j, "target"] = 0
    zsl_df_train.loc[j, "task"] = train_df.task.iloc[i]
    zsl_df_train.loc[j, "actual category"] = train_df.category.iloc[i]
    j += 1

zsl_df_train = zsl_df_train.sample(frac=1).reset_index(drop=True)

zsl_df_train.head()

zsl_df_train.tail()

zsl_df_val = pd.DataFrame()
j = 0

for i in tqdm(range(val_df.shape[0])):
    zsl_df_val.loc[j,'text'] = val_df.text.iloc[i] #df.category.iloc[i].lower() + " text - " + df.text.iloc[i]
    zsl_df_val.loc[j, "category"] = val_df.category.iloc[i]
    zsl_df_val.loc[j, "target"] = 1
    zsl_df_val.loc[j, "task"] = val_df.task.iloc[i]
    zsl_df_val.loc[j, "actual category"] = val_df.category.iloc[i]
    j += 1
    random_label = random.sample(list(set(val_df[val_df.task == val_df.task.iloc[i]].category.unique().tolist()) - set([val_df.category.iloc[i]])),1)
    zsl_df_val.loc[j,'text'] = val_df.text.iloc[i] #random_label[0].lower() + " text - " + df.text.iloc[i]
    zsl_df_val.loc[j, "category"] = random_label #df.category.iloc[i]
    zsl_df_val.loc[j, "target"] = 0
    zsl_df_val.loc[j, "task"] = val_df.task.iloc[i]
    zsl_df_val.loc[j, "actual category"] = val_df.category.iloc[i]
    j += 1

zsl_df_val = zsl_df_val.sample(frac=1).reset_index(drop=True)

zsl_df_test = pd.DataFrame()
j = 0

for i in tqdm(range(test_df.shape[0])):
    zsl_df_test.loc[j,'text'] = test_df.text.iloc[i] #df.category.iloc[i].lower() + " text - " + df.text.iloc[i]
    zsl_df_test.loc[j, "category"] = test_df.category.iloc[i]
    zsl_df_test.loc[j, "target"] = 1
    zsl_df_test.loc[j, "task"] = test_df.task.iloc[i]
    zsl_df_test.loc[j, "actual category"] = test_df.category.iloc[i]
    j += 1
    random_label = random.sample(list(set(test_df[test_df.task == test_df.task.iloc[i]].category.unique().tolist()) - set([test_df.category.iloc[i]])),1)
    zsl_df_test.loc[j,'text'] = test_df.text.iloc[i] #random_label[0].lower() + " text - " + df.text.iloc[i]
    zsl_df_test.loc[j, "category"] = random_label #df.category.iloc[i]
    zsl_df_test.loc[j, "target"] = 0
    zsl_df_test.loc[j, "task"] = test_df.task.iloc[i]
    zsl_df_test.loc[j, "actual category"] = test_df.category.iloc[i]
    j += 1

zsl_df_test = zsl_df_test.sample(frac=1).reset_index(drop=True)

zsl_df_train['category_id'] = zsl_df_train.category.apply(lambda x: inv_category_dict[x])
zsl_df_val['category_id'] = zsl_df_val.category.apply(lambda x: inv_category_dict[x])
zsl_df_test['category_id'] = zsl_df_test.category.apply(lambda x: inv_category_dict[x])

zsl_df_train.head()

zsl_df_train.target = zsl_df_train.target.astype(int)
zsl_df_val.target = zsl_df_val.target.astype(int)
zsl_df_test.target = zsl_df_test.target.astype(int)
zsl_df_train.target.value_counts(),zsl_df_val.target.value_counts(),zsl_df_test.target.value_counts()

model_save_dir = args.model_save_path

try:
    os.makedirs(model_save_dir)
except OSError:
    pass

model_save_dir

zsl_df_train.to_csv(os.path.join(args.model_save_path, 'train.csv'), index=False, sep='\t')
zsl_df_val.to_csv(os.path.join(args.model_save_path, 'valid.csv'), index=False, sep='\t')
zsl_df_test.to_csv(os.path.join(args.model_save_path, 'test.csv'), index=False, sep='\t')

"""### Learn tokenizer"""

data.custom_tokenizers.custom_wp_tokenizer(train_df.text.values, args.model_save_path, args.model_save_path)
tokenizer = BertTokenizer.from_pretrained(args.model_save_path)
bert_tokenizer = BertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1)
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1)

word_tokenizer.fit_on_texts(train_df.text.values.tolist()) #+ train_df.category.values.tolist()
char_tokenizer.fit_on_texts(train_df.text.values)

transformer_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_train, 'text', tokenizer, args.max_char_len)

word_train_inputs = word_tokenizer.texts_to_sequences(zsl_df_train.text.values)
word_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs, maxlen=args.max_text_len, padding='post')

subword_train_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(zsl_df_train.text.values)])

char_train_inputs = char_tokenizer.texts_to_sequences(zsl_df_train.text.values)
char_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_train_inputs, maxlen=args.max_char_len, padding='post')

train_outputs = data.data_utils.compute_output_arrays(zsl_df_train, 'target')[:,np.newaxis]

transformer_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_val, 'text', tokenizer, args.max_char_len)

word_val_inputs = word_tokenizer.texts_to_sequences(zsl_df_val.text.values)
word_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_val_inputs, maxlen=args.max_text_len, padding='post')

subword_val_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(zsl_df_val.text.values)])

char_val_inputs = char_tokenizer.texts_to_sequences(zsl_df_val.text.values)
char_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_val_inputs, maxlen=args.max_char_len, padding='post')

val_outputs = data.data_utils.compute_output_arrays(zsl_df_val, 'target')[:,np.newaxis]

transformer_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_test, 'text', tokenizer, args.max_char_len)

word_test_inputs = word_tokenizer.texts_to_sequences(zsl_df_test.text.values)
word_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs, maxlen=args.max_text_len, padding='post')

subword_test_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(zsl_df_test.text.values)])

char_test_inputs = char_tokenizer.texts_to_sequences(zsl_df_test.text.values)
char_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_test_inputs, maxlen=args.max_char_len, padding='post')

test_outputs = data.data_utils.compute_output_arrays(zsl_df_test, 'target')[:,np.newaxis]

#train_outputs = tf.keras.utils.to_categorical(train_outputs, \
#                                                    num_classes=train_df.category.nunique())
#val_outputs = tf.keras.utils.to_categorical(val_outputs, \
#                                                    num_classes=train_df.category.nunique())
#test_outputs = tf.keras.utils.to_categorical(test_outputs, \
#                                                    num_classes=train_df.category.nunique())

tfidf1 = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_df=.6,min_df=2, max_features=5000)
tfidf2 = TfidfVectorizer(analyzer='char_wb',ngram_range=(1,3), max_df=.6,min_df=2, max_features=5000)

tfidf1.fit(train_df.text)
tfidf2.fit(train_df.text)

train_tfidf = np.hstack([tfidf1.transform(zsl_df_train.text).toarray(),tfidf2.transform(zsl_df_train.text).toarray()])
val_tfidf = np.hstack([tfidf1.transform(zsl_df_val.text).toarray(),tfidf2.transform(zsl_df_val.text).toarray()])
test_tfidf = np.hstack([tfidf1.transform(zsl_df_test.text).toarray(),tfidf2.transform(zsl_df_test.text).toarray()])


print (transformer_train_inputs.shape, subword_train_inputs.shape, word_train_inputs.shape, char_train_inputs.shape, \
       train_tfidf.shape, train_outputs.shape)
print (transformer_val_inputs.shape, subword_val_inputs.shape, word_val_inputs.shape, char_val_inputs.shape, \
       val_tfidf.shape, val_outputs.shape)
print (transformer_test_inputs.shape, subword_test_inputs.shape, word_test_inputs.shape, char_test_inputs.shape, \
       test_tfidf.shape, test_outputs.shape)

word_train_inputs_category = word_tokenizer.texts_to_sequences(zsl_df_train.category.values)
word_train_inputs_category = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs_category, maxlen=1, padding='post')

subword_train_inputs_category = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, 1, args.max_word_char_len) \
                        for text in tqdm(zsl_df_train.category.values)])

word_val_inputs_category = word_tokenizer.texts_to_sequences(zsl_df_val.category.values)
word_val_inputs_category = tf.keras.preprocessing.sequence.pad_sequences(word_val_inputs_category, maxlen=1, padding='post')

subword_val_inputs_category = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, 1, args.max_word_char_len) \
                        for text in tqdm(zsl_df_val.category.values)])

word_test_inputs_category = word_tokenizer.texts_to_sequences(zsl_df_test.category.values)
word_test_inputs_category = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs_category, maxlen=1, padding='post')

subword_test_inputs_category = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, 1, args.max_word_char_len) \
                        for text in tqdm(zsl_df_test.category.values)])

print (word_train_inputs_category.shape, word_val_inputs_category.shape, word_test_inputs_category.shape, \
      subword_train_inputs_category.shape, subword_val_inputs_category.shape, subword_test_inputs_category.shape)

bert_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_train, 'text', bert_tokenizer, args.max_char_len)

bert_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_val, 'text', bert_tokenizer, args.max_char_len)

bert_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_test, 'text', bert_tokenizer, args.max_char_len)

bert_category_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_train, 'category', bert_tokenizer, args.max_char_len)

bert_category_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_val, 'category', bert_tokenizer, args.max_char_len)

bert_category_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(zsl_df_test, 'category', bert_tokenizer, args.max_char_len)

from transformers import TFBertModel

bert_backbone = TFBertModel.from_pretrained("distilbert-base-multilingual-cased")

"""### Modeling"""

n_words = len(word_tokenizer.word_index)+1
n_chars = len(char_tokenizer.word_index)+1
n_subwords = tokenizer.vocab_size
tfidf_shape = train_tfidf.shape[1]
n_out = 1 #train_df.category.nunique()

import tensorflow as tf
import numpy as np
from src.models.layers import *

'''
def BERT_ZSL(backbone, n_out, seq_output=False, vectorizer_shape=None,max_char_len=100,n_units=128):
    
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    wpe_inputs_category = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    pooling = tf.keras.layers.GlobalAveragePooling1D()
    drop1 = tf.keras.layers.Dropout(0.2)
    dense1 = tf.keras.layers.Dense(n_units)
    dense2 = tf.keras.layers.Dense(n_units)
    drop2 = tf.keras.layers.Dropout(0.2)
    dot1 = tf.keras.layers.Dot(axes=1)
    if n_out > 1:
        dense3 = tf.keras.layers.Dense(n_out, activation='softmax')
    else:
        dense3 = tf.keras.layers.Dense(n_out, activation='sigmoid')

    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
    else:
        tfidf = None
    def get_emb(subword_inputs_):
        x = backbone(subword_inputs_)[0]

        if seq_output == False:
            x = pooling(x)
        x = drop1(x)
        
        return x
    
    x = get_emb(wpe_inputs)
    if vectorizer_shape:
        x = dense1(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = dense1(x)

    x = drop2(x)
        
    x_category = get_emb(wpe_inputs_category)
    x_category = dense2(x_category)
    x_category = drop2(x_category)
    
    x = dot1([x, x_category])
    out = dense3(x)
    #out = tf.keras.layers.Activation('sigmoid')(x) #

    if vectorizer_shape:
        model = tf.keras.models.Model([wpe_inputs,wpe_inputs_category, tfidf], out)
    else:
        model = tf.keras.models.Model([wpe_inputs,wpe_inputs_category], out)

    return model
'''

def HIT(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    x = embedding_layer(char_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units,outer_attention=False))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    
    word_inputs_category = tf.keras.layers.Input((1,), dtype=tf.int32)
    subword_inputs_category = tf.keras.layers.Input((1,max_word_char_len,), dtype=tf.int32)
    
    char_model = tf.keras.layers.TimeDistributed(char_model)
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    word_emb_layer = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)
    pooling = tf.keras.layers.GlobalAveragePooling1D()
    drop1 = tf.keras.layers.Dropout(0.2)
    dense1 = tf.keras.layers.Dense(n_units)
    dense2 = tf.keras.layers.Dense(n_units)
    drop2 = tf.keras.layers.Dropout(0.2)
    dot1 = tf.keras.layers.Dot(axes=1)
    if n_out > 1:
        dense3 = tf.keras.layers.Dense(n_out, activation='softmax')
    else:
        dense3 = tf.keras.layers.Dense(n_out, activation='sigmoid')

    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
    else:
        tfidf = None
    def get_emb(word_inputs_, subword_inputs_):
        x = char_model(subword_inputs_)

        position = embedding_layer(word_inputs_)

        x = x + position

        word_embbeding = word_emb_layer(word_inputs)

        x = x + word_embbeding

        for i in range(n_layers):
            x = transformer_blocks[i](x)

        if seq_output == False:
            x = pooling(x)
        x = drop1(x)
        
        return x
    
    x = get_emb(word_inputs, subword_inputs)
    if vectorizer_shape:
        x = dense1(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = dense1(x)

    x = drop2(x)
        
    x_category = get_emb(word_inputs_category, subword_inputs_category)
    x_category = dense2(x_category)
    x_category = drop2(x_category)
    
    x = dot1([x, x_category])
    out = dense3(x)
    #out = tf.keras.layers.Activation('sigmoid')(x) #

    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,word_inputs_category, subword_inputs_category, tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,word_inputs_category, subword_inputs_category], out)

    return model

def HIT_outer(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    x = embedding_layer(char_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units,outer_attention=True))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    
    word_inputs_category = tf.keras.layers.Input((1,), dtype=tf.int32)
    subword_inputs_category = tf.keras.layers.Input((1,max_word_char_len,), dtype=tf.int32)
    
    char_model = tf.keras.layers.TimeDistributed(char_model)
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    word_emb_layer = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)
    pooling = tf.keras.layers.GlobalAveragePooling1D()
    drop1 = tf.keras.layers.Dropout(0.2)
    dense1 = tf.keras.layers.Dense(n_units)
    dense2 = tf.keras.layers.Dense(n_units)
    drop2 = tf.keras.layers.Dropout(0.2)
    dot1 = tf.keras.layers.Dot(axes=1)
    if n_out > 1:
        dense3 = tf.keras.layers.Dense(n_out, activation='softmax')
    else:
        dense3 = tf.keras.layers.Dense(n_out, activation='sigmoid')

    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
    else:
        tfidf = None
        
    def get_emb(word_inputs_, subword_inputs_):
        x = char_model(subword_inputs_)

        position = embedding_layer(word_inputs_)

        x = x + position

        word_embbeding = word_emb_layer(word_inputs)

        x = x + word_embbeding

        for i in range(n_layers):
            x = transformer_blocks[i](x)

        if seq_output == False:
            x = pooling(x)
        x = drop1(x)
        
        return x
    
    x = get_emb(word_inputs, subword_inputs)
    if vectorizer_shape:
        x = dense1(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = dense1(x)

    x = drop2(x)
        
    x_category = get_emb(word_inputs_category, subword_inputs_category)
    x_category = dense2(x_category)
    x_category = drop2(x_category)
    
    x = dot1([x, x_category])
    
    out = dense3(x)
    #out = tf.keras.layers.Activation('sigmoid')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,word_inputs_category, subword_inputs_category, tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,word_inputs_category, subword_inputs_category], out)

    return model

'''
all_models = {BERT_ZSL.__name__: BERT_ZSL}

_has_wandb = False

import time

if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

for model_name, model_ in all_models.items():
    
    for loss in ['ce']:
        
        for use_features in [False]:
            
            if use_features == False:
                model = model_(bert_backbone, n_out=n_out, seq_output=False, vectorizer_shape=None,max_char_len=args.max_char_len,n_units=args.n_units)
            else:
                model = model_(bert_backbone, n_out=n_out, seq_output=False, vectorizer_shape=tfidf_shape,max_char_len=args.max_char_len,n_units=args.n_units)
            
            if use_features == True:
                print ("Running {} with features for {} loss".format(model_name, loss))
            else:
                print ("Running {} without features for {} loss".format(model_name, loss))

            print (model.summary())

            if loss == 'focal':
                model.compile(loss=models.utils.categorical_focal_loss(alpha=1), optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) #binary_crossentropy
            elif loss == 'ce':
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) 

            lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, \
                                                  patience=args.lr_schedule_round, verbose=1, mode='auto', min_lr=0.000001)
            config = {
                  'text_max_len': args.max_text_len,
                  'char_max_len': args.max_char_len,
                  'word_char_max_len': args.max_word_char_len,
                  'n_units': args.n_units,
                  'emb_dim': args.emb_dim,
                  'n_layers': args.n_layers,
                  'epochs': args.epochs,
                  "learning_rate": args.lr,
                  "model_name": model_name,
                  "loss": loss,
                  "use_features": use_features
                }

            if use_features == True:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_with_features.h5'.format(model_name, config['loss']))
            else:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_without_features.h5'.format(model_name, config['loss']))

            #f1callback = models.utils.F1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, val_tfidf],\
            #                          val_outputs, \
            #                          filename=model_save_path, \
            #                          patience=args.early_stopping_rounds)
            
            early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', min_delta=0, patience=args.early_stopping_rounds, verbose=0,restore_best_weights=True,
                        mode='auto')
            
            K.clear_session()
            
            if use_features == True:
                if _has_wandb and args.wandb_logging:
                    wandb.init(project='hindi_zsl',config=config)
                    model.fit([bert_train_inputs, bert_category_train_inputs, train_tfidf], train_outputs, \
                          validation_data=([bert_val_inputs, bert_category_val_inputs, val_tfidf], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                else:
                    model.fit([bert_train_inputs, bert_category_train_inputs, train_tfidf], train_outputs, \
                          validation_data=([bert_val_inputs, bert_category_val_inputs, val_tfidf], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=1)
            else:
                if _has_wandb and args.wandb_logging:
                    wandb.init(project='hindi_zsl',config=config)
                    model.fit([bert_train_inputs, bert_category_train_inputs], train_outputs, \
                          validation_data=([bert_val_inputs, bert_category_val_inputs], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                else:
                    model.fit([bert_train_inputs, bert_category_train_inputs], train_outputs, \
                          validation_data=([bert_val_inputs, bert_category_val_inputs], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=1)

            
            #try:
            #model.load_weights(model_save_path)

            start = time.time()
            if use_features == True:
                test_pred = model.predict([bert_test_inputs, bert_category_test_inputs, test_tfidf])
            else:
                test_pred = model.predict([bert_test_inputs, bert_category_test_inputs])
            end = time.time()

            print ("time taken {} seconds".format(end-start))
            
            zsl_df_test['pred_proba'] = test_pred
            zsl_df_test['pred_target'] = zsl_df_test.pred_proba.round().astype(int)
            #print (zsl_df_test[zsl_df_test.target == 1].groupby(['category'])['pred_target'].value_counts(normalize=True))
            #print (zsl_df_test[zsl_df_test.target == 0].groupby(['category'])['pred_target'].value_counts(normalize=True))
            zsl_df_test_uniq = zsl_df_test.groupby(['text'])['pred_proba'].max().reset_index(drop=False)
            zsl_df_test_ = pd.merge(zsl_df_test, zsl_df_test_uniq,how='inner')
            print (zsl_df_test_.groupby(['task']).apply(lambda x: accuracy_score(x.category, x['actual category'])), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='weighted')), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: precision_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: recall_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: confusion_matrix(x.category, x['actual category'])))
            #break

            #results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)
            #except:
            #    pass
'''

#from src.models.models import *

#all_models = {HIT_without_words.__name__:HIT_without_words, CS_ELMO_without_words.__name__: CS_ELMO_without_words, Transformer.__name__: Transformer, HAN.__name__: HAN}

all_models = {HIT_outer.__name__:HIT_outer, HIT.__name__: HIT} #, CS_ELMO.__name__: CS_ELMO, \
                  #Transformer.__name__: Transformer, HAN.__name__: HAN, CMSA.__name__: CMSA, WLSTM.__name__: WLSTM}


import time

if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

for model_name, model_ in all_models.items():
    
    for loss in ['ce']:
        
        for use_features in [False, True]:
            
            if use_features == False:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=n_out,max_word_char_len=args.max_word_char_len,\
                                             max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                             n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
            else:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords,n_out=n_out,vectorizer_shape=tfidf_shape, max_word_char_len=args.max_word_char_len,\
                                             max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                             n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
            
            if use_features == True:
                print ("Running {} with features for {} loss".format(model_name, loss))
            else:
                print ("Running {} without features for {} loss".format(model_name, loss))

            #print (model.summary())

            if loss == 'focal':
                model.compile(loss=models.utils.categorical_focal_loss(alpha=1), optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) #binary_crossentropy
            elif loss == 'ce':
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) 

            lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, \
                                                  patience=args.lr_schedule_round, verbose=1, mode='auto', min_lr=0.000001)
            config = {
                  'text_max_len': args.max_text_len,
                  'char_max_len': args.max_char_len,
                  'word_char_max_len': args.max_word_char_len,
                  'n_units': args.n_units,
                  'emb_dim': args.emb_dim,
                  'n_layers': args.n_layers,
                  'epochs': args.epochs,
                  "learning_rate": args.lr,
                  "model_name": model_name,
                  "loss": loss,
                  "use_features": use_features
                }

            if use_features == True:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_with_features.h5'.format(model_name, config['loss']))
            else:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_without_features.h5'.format(model_name, config['loss']))

            #f1callback = models.utils.F1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, val_tfidf],\
            #                          val_outputs, \
            #                          filename=model_save_path, \
            #                          patience=args.early_stopping_rounds)
            
            early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', min_delta=0, patience=args.early_stopping_rounds, verbose=0,restore_best_weights=True,
                        mode='auto')
            
            K.clear_session()
            
            if use_features == True:
                if _has_wandb and args.wandb_logging:
                    wandb.init(project='hindi_zsl',config=config)
                    model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, word_train_inputs_category, subword_train_inputs_category, train_tfidf], train_outputs, \
                          validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category, val_tfidf], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                else:
                    model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs,word_train_inputs_category, subword_train_inputs_category, train_tfidf], train_outputs, \
                          validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category, val_tfidf], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=1)
            else:
                if _has_wandb and args.wandb_logging:
                    wandb.init(project='hindi_zsl',config=config)
                    model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs,word_train_inputs_category, subword_train_inputs_category], train_outputs, \
                          validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                else:
                    model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs,word_train_inputs_category, subword_train_inputs_category], train_outputs, \
                          validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs,word_val_inputs_category, subword_val_inputs_category], val_outputs), \
                              epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=1)

            
            #try:
            #model.load_weights(model_save_path)

            start = time.time()
            if use_features == True:
                test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, word_test_inputs_category, subword_test_inputs_category, test_tfidf])
            else:
                test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, word_test_inputs_category, subword_test_inputs_category])
            end = time.time()

            print ("time taken {} seconds".format(end-start))
            
            zsl_df_test['pred_proba'] = test_pred
            zsl_df_test['pred_target'] = zsl_df_test.pred_proba.round().astype(int)
            #print (zsl_df_test[zsl_df_test.target == 1].groupby(['category'])['pred_target'].value_counts(normalize=True))
            #print (zsl_df_test[zsl_df_test.target == 0].groupby(['category'])['pred_target'].value_counts(normalize=True))
            zsl_df_test_uniq = zsl_df_test.groupby(['text'])['pred_proba'].max().reset_index(drop=False)
            zsl_df_test_ = pd.merge(zsl_df_test, zsl_df_test_uniq,how='inner')
            print (zsl_df_test_.groupby(['task']).apply(lambda x: accuracy_score(x.category, x['actual category'])), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='weighted')), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: precision_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: recall_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: confusion_matrix(x.category, x['actual category'])))
            #break

            #results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)
            #except:
            #    pass

import time

if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

for task in zsl_df_train.task.unique():
  train_idx = zsl_df_train[zsl_df_train.task != task].index
  print ("Number of training instances {}".format(len(train_idx)))
  for model_name, model_ in all_models.items():
      
      for loss in ['ce']:
          
          for use_features in [False, True]:
              
              if use_features == False:
                  model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=n_out,max_word_char_len=args.max_word_char_len,\
                                              max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                              n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
              else:
                  model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords,n_out=n_out,vectorizer_shape=tfidf_shape, max_word_char_len=args.max_word_char_len,\
                                              max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                              n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
              
              if use_features == True:
                  print ("Running {} with features for {} loss".format(model_name, loss))
              else:
                  print ("Running {} without features for {} loss".format(model_name, loss))

              #print (model.summary())

              if loss == 'focal':
                  model.compile(loss=models.utils.categorical_focal_loss(alpha=1), optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) #binary_crossentropy
              elif loss == 'ce':
                  model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr), metrics=['accuracy']) 

              lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, \
                                                    patience=args.lr_schedule_round, verbose=1, mode='auto', min_lr=0.000001)
              config = {
                    'text_max_len': args.max_text_len,
                    'char_max_len': args.max_char_len,
                    'word_char_max_len': args.max_word_char_len,
                    'n_units': args.n_units,
                    'emb_dim': args.emb_dim,
                    'n_layers': args.n_layers,
                    'epochs': args.epochs,
                    "learning_rate": args.lr,
                    "model_name": model_name,
                    "loss": loss,
                    "use_features": use_features
                  }

              if use_features == True:
                  model_save_path = os.path.join(args.model_save_path, '{}_{}_with_features.h5'.format(model_name, config['loss']))
              else:
                  model_save_path = os.path.join(args.model_save_path, '{}_{}_without_features.h5'.format(model_name, config['loss']))

              #f1callback = models.utils.F1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, val_tfidf],\
              #                          val_outputs, \
              #                          filename=model_save_path, \
              #                          patience=args.early_stopping_rounds)
              
              early_stop = tf.keras.callbacks.EarlyStopping(
                          monitor='val_loss', min_delta=0, patience=args.early_stopping_rounds, verbose=0,restore_best_weights=True,
                          mode='auto')
              
              K.clear_session()
              
              if use_features == True:
                  if _has_wandb and args.wandb_logging:
                      wandb.init(project='hindi_zsl',config=config)
                      model.fit([word_train_inputs[train_idx], char_train_inputs[train_idx], subword_train_inputs[train_idx], transformer_train_inputs[train_idx], word_train_inputs_category[train_idx], subword_train_inputs_category[train_idx], train_tfidf[train_idx]], train_outputs[train_idx], \
                            validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category, val_tfidf], val_outputs), \
                                epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                  else:
                      model.fit([word_train_inputs[train_idx], char_train_inputs[train_idx], subword_train_inputs[train_idx], transformer_train_inputs[train_idx], word_train_inputs_category[train_idx], subword_train_inputs_category[train_idx], train_tfidf[train_idx]], train_outputs[train_idx], \
                            validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category, val_tfidf], val_outputs), \
                                epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=2)
              else:
                  if _has_wandb and args.wandb_logging:
                      wandb.init(project='hindi_zsl',config=config)
                      model.fit([word_train_inputs[train_idx], char_train_inputs[train_idx], subword_train_inputs[train_idx], transformer_train_inputs[train_idx],word_train_inputs_category[train_idx], subword_train_inputs_category[train_idx]], train_outputs[train_idx], \
                            validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category], val_outputs), \
                                epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop, WandbCallback()], verbose=2)
                  else:
                      model.fit([word_train_inputs[train_idx], char_train_inputs[train_idx], subword_train_inputs[train_idx], transformer_train_inputs[train_idx],word_train_inputs_category[train_idx], subword_train_inputs_category[train_idx]], train_outputs[train_idx], \
                            validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, word_val_inputs_category, subword_val_inputs_category], val_outputs), \
                                epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, early_stop], verbose=2)

              
              #try:
              #model.load_weights(model_save_path)

              start = time.time()
              if use_features == True:
                  test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, word_test_inputs_category, subword_test_inputs_category, test_tfidf])
              else:
                  test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, word_test_inputs_category, subword_test_inputs_category])
              end = time.time()

              print ("time taken {} seconds".format(end-start))
              
              zsl_df_test['pred_proba'] = test_pred
              zsl_df_test['pred_target'] = zsl_df_test.pred_proba.round().astype(int)
              #print (zsl_df_test[zsl_df_test.target == 1].groupby(['category'])['pred_target'].value_counts(normalize=True))
              #print (zsl_df_test[zsl_df_test.target == 0].groupby(['category'])['pred_target'].value_counts(normalize=True))
              zsl_df_test_uniq = zsl_df_test.groupby(['text'])['pred_proba'].max().reset_index(drop=False)
              zsl_df_test_ = pd.merge(zsl_df_test, zsl_df_test_uniq,how='inner')
              print (zsl_df_test_.groupby(['task']).apply(lambda x: accuracy_score(x.category, x['actual category'])), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='weighted')), zsl_df_test_.groupby(['task']).apply(lambda x: f1_score(x.category, x['actual category'],average='macro')), zsl_df_test_.groupby(['task']).apply(lambda x: confusion_matrix(x.category, x['actual category'])))
              #break

              #results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)
              #except:
              #    pass

zsl_df_train.task.unique()

zsl_df_train.task.value_counts()
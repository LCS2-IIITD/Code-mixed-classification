from __future__ import absolute_import

import sys
import os

sys.path.append('./drive/My Drive/CMC/')


# In[2]:


get_ipython().system('wandb login')


# In[3]:


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

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

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

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

from src import data, models

pd.options.display.max_colwidth = -1


# In[4]:


print (_has_wandb)


# In[5]:


parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

parser.add_argument('--train_data', type=str, default='../data/NER/NER Hindi English Code Mixed Tweets.tsv', required=False,
                    help='train data')
#parser.add_argument('--train_data', type=str, default='./drive/My Drive/CMC/data/NER/NER Hindi English Code Mixed Tweets.tsv', required=False,
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

parser.add_argument('--max_text_len', type=int, default=30, required=False,
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

parser.add_argument('--epochs', type=int, default=500, required=False,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=.001, required=False,
                    help='learning rate')
parser.add_argument('--early_stopping_rounds', type=int, default=50, required=False,
                    help='number of epochs for early stopping')
parser.add_argument('--lr_schedule_round', type=int, default=30, required=False,
                    help='number of epochs for learning rate scheduling')

parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                    help='eval batch size')

parser.add_argument('--model_save_path', type=str, default='../models/hindi_ner/', required=False,
                    help='model')
#parser.add_argument('--model_save_path', type=str, default='./drive/My Drive/CMC/models/hindi_ner/', required=False,
#                    help='model')

parser.add_argument('--wandb_logging', type=bool, default=True, required=False,
                    help='wandb logging needed')

parser.add_argument('--seed', type=int, default=42, required=False,
                    help='seed')


args, _ = parser.parse_known_args()


# In[6]:


tf.random.set_seed(args.seed)
np.random.seed(args.seed)


# In[7]:


pos_data = data.data_utils.CoNLLSeqData(filepath=args.train_data,label_index=2)

df = pd.DataFrame()
df['sentence'] = pos_data.sentence
df['text'] = [" ".join(i) for i in pos_data.words]
df['category'] = [" ".join(i) for i in pos_data.labels]

kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
for train_index, test_index in kf.split(df.text):
    break

train_df = df.iloc[train_index]
kf2 = KFold(n_splits=2, shuffle=True, random_state=args.seed)
for val_index, test_index in kf2.split(df.iloc[test_index].text):
    break

val_df = df.iloc[val_index]
test_df = df.iloc[test_index]

model_save_dir = args.model_save_path

try:
    os.makedirs(model_save_dir)
except OSError:
    pass


# ### Learn tokenizer

# In[12]:


#data.custom_tokenizers.custom_wp_tokenizer(train_df.text.values, args.model_save_path, args.model_save_path)
tokenizer = BertTokenizer.from_pretrained(args.model_save_path)


# In[13]:


word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1, filters='', lower=False)
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1, lower=False)
tag_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ', filters='', lower=False)

word_tokenizer.fit_on_texts(train_df.text.values)
char_tokenizer.fit_on_texts(train_df.text.values)
tag_tokenizer.fit_on_texts(train_df.category.values)


# In[14]:


label2idx = tag_tokenizer.word_index
idx2label = tag_tokenizer.index_word


# In[15]:


transformer_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(train_df, 'text', tokenizer, args.max_char_len)

word_train_inputs = word_tokenizer.texts_to_sequences(train_df.text.values)
word_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs, maxlen=args.max_text_len)

subword_train_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len)\
                         for text in tqdm(train_df.text.values)])

char_train_inputs = char_tokenizer.texts_to_sequences(train_df.text.values)
char_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_train_inputs, maxlen=args.max_char_len)

train_outputs = tag_tokenizer.texts_to_sequences(train_df.category.values)
train_outputs = tf.keras.preprocessing.sequence.pad_sequences(train_outputs, maxlen=args.max_text_len)

train_outputs2 = tag_tokenizer.texts_to_sequences(train_df.category.values)
train_outputs2 = tf.keras.preprocessing.sequence.pad_sequences(train_outputs2, maxlen=args.max_char_len)

transformer_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(val_df, 'text', tokenizer, args.max_char_len)

word_val_inputs = word_tokenizer.texts_to_sequences(val_df.text.values)
word_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_val_inputs, maxlen=args.max_text_len)

subword_val_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(val_df.text.values)])

char_val_inputs = char_tokenizer.texts_to_sequences(val_df.text.values)
char_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_val_inputs, maxlen=args.max_char_len)

val_outputs = tag_tokenizer.texts_to_sequences(val_df.category.values)
val_outputs = tf.keras.preprocessing.sequence.pad_sequences(val_outputs, maxlen=args.max_text_len)

val_outputs2 = tag_tokenizer.texts_to_sequences(val_df.category.values)
val_outputs2 = tf.keras.preprocessing.sequence.pad_sequences(val_outputs2, maxlen=args.max_char_len)

transformer_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(test_df, 'text', tokenizer, args.max_char_len)

word_test_inputs = word_tokenizer.texts_to_sequences(test_df.text.values)
word_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs, maxlen=args.max_text_len)

subword_test_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(test_df.text.values)])

char_test_inputs = char_tokenizer.texts_to_sequences(test_df.text.values)
char_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_test_inputs, maxlen=args.max_char_len)

test_outputs = tag_tokenizer.texts_to_sequences(test_df.category.values)
test_outputs = tf.keras.preprocessing.sequence.pad_sequences(test_outputs, maxlen=args.max_text_len)

test_outputs2 = tag_tokenizer.texts_to_sequences(test_df.category.values)
test_outputs2 = tf.keras.preprocessing.sequence.pad_sequences(test_outputs2, maxlen=args.max_char_len)

train_outputs = tf.keras.utils.to_categorical(train_outputs, num_classes=len(label2idx)+1)
val_outputs = tf.keras.utils.to_categorical(val_outputs, num_classes=len(label2idx)+1)
test_outputs = tf.keras.utils.to_categorical(test_outputs,num_classes=len(label2idx)+1)

train_outputs2 = tf.keras.utils.to_categorical(train_outputs2,num_classes=len(label2idx)+1)
val_outputs2 = tf.keras.utils.to_categorical(val_outputs2,  num_classes=len(label2idx)+1)
test_outputs2 = tf.keras.utils.to_categorical(test_outputs2, num_classes=len(label2idx)+1)

tfidf1 = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_df=.6,min_df=2)
tfidf2 = TfidfVectorizer(analyzer='char_wb',ngram_range=(1,3), max_df=.6,min_df=2)

tfidf1.fit(train_df.text)
tfidf2.fit(train_df.text)

train_tfidf = np.hstack([tfidf1.transform(train_df.text).toarray(),tfidf2.transform(train_df.text).toarray()])
val_tfidf = np.hstack([tfidf1.transform(val_df.text).toarray(),tfidf2.transform(val_df.text).toarray()])
test_tfidf = np.hstack([tfidf1.transform(test_df.text).toarray(),tfidf2.transform(test_df.text).toarray()])


# In[16]:


n_words = len(word_tokenizer.word_index)+1
n_chars = len(char_tokenizer.word_index)+1
n_subwords = tokenizer.vocab_size
tfidf_shape = train_tfidf.shape[1]
n_out = len(label2idx)+1


# In[17]:


label2idx


# In[18]:


label2idx['PAD'] = 0
idx2label[0] = 'PAD'
idx2label = {value-1: key for (key,value) in label2idx.items()}

# In[27]:


from src.models.models import *
#from src.models.layers import *
all_models = {HIT_outer.__name__:HIT_outer,HIT.__name__: HIT}

#all_models = {HIT.__name__:HIT}


# In[28]:


if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

for model_name, model_ in all_models.items():
    
    for loss in ['ce','focal']:
        
        model = model_(word_vocab_size=n_words,char_vocab_size=n_chars, wpe_vocab_size=n_subwords, n_out=n_out,seq_output=True,max_word_char_len=args.max_word_char_len,                                          max_text_len=args.max_text_len, max_char_len=args.max_char_len,                                          n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
        
        print ("Running {} without features for {} loss".format(model_name, loss))

        print (model.summary())

        if loss == 'focal':
            model.compile(loss=models.utils.categorical_focal_loss(alpha=1), optimizer='adam', metrics=['accuracy', models.utils.f1_keras]) #binary_crossentropy
        elif loss == 'ce':
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', models.utils.f1_keras]) 

        lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7,                                               patience=args.lr_schedule_round, verbose=1, mode='auto', min_lr=0.000001)
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
              "use_features": False
            }

        model_save_path = os.path.join(args.model_save_path, '{}_{}.h5'.format(model_name, config['loss']))

        if model_name != 'Transformer':
          f1callback = models.utils.SeqF1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf],                                      val_outputs,                                       filename=model_save_path,                                       patience=args.early_stopping_rounds)

          K.clear_session()
          
          
          if _has_wandb and args.wandb_logging:
              wandb.init(project='hindi_ner',config=config)
              model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs, \
                    validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs), \
                        epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback, WandbCallback()], verbose=2)
          else:
              model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs, \
                    validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs), \
                        epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback], verbose=2)
          

          model.load_weights(model_save_path)

          model.load_weights(model_save_path)

          test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, test_tfidf])

          test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs])

          test_pred = test_pred[:,:,1:]

          report = flat_classification_report([[idx2label[j] for j in i] for i in test_outputs[:,:,1:].argmax(-1)], \
                                            [[idx2label[j] for j in i] for i in test_pred.argmax(-1)])

          f1 = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='weighted')

          results.loc[index,'config'] = str(config)
          results.loc[index, 'weighted_f1'] = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='weighted')
          results.loc[index, 'macro_f1'] = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='macro')

          results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)

        else:
          f1callback = models.utils.SeqF1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf],                                      val_outputs2,                                       filename=model_save_path,                                       patience=args.early_stopping_rounds)

          K.clear_session()
          
          
          if _has_wandb and args.wandb_logging:
              wandb.init(project='hindi_ner',config=config)
              model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs2, \
                    validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs2), \
                        epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback, WandbCallback()], verbose=2)
          else:
              model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs2, \
                    validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs2), \
                        epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback], verbose=2)
          

          model.load_weights(model_save_path)

          test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, test_tfidf])

          test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs])

          test_pred = test_pred[:,:,1:]

          report = flat_classification_report([[idx2label[j] for j in i] for i in test_outputs[:,:,1:].argmax(-1)], \
                                            [[idx2label[j] for j in i] for i in test_pred.argmax(-1)])

          f1 = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='weighted')

          results.loc[index,'config'] = str(config)
          results.loc[index, 'weighted_f1'] = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='weighted')
          results.loc[index, 'macro_f1'] = flat_f1_score(test_outputs[:,:,1:].argmax(-1), test_pred.argmax(-1), average='macro')
        
          results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)

        index += 1

        print (report, f1)


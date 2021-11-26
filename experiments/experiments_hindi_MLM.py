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

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src import data, models

pd.options.display.max_colwidth = -1

print (_has_wandb)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

parser.add_argument('--train_data', type=str, default='../data/hindi_sentiment/IIITH_Codemixed.txt', required=False,
                    help='train data')
#parser.add_argument('--train_data', type=str, default='../data/IIITH_Codemixed.txt', required=False,
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

parser.add_argument('--epochs', type=int, default=500, required=False,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=.001, required=False,
                    help='learning rate')
parser.add_argument('--early_stopping_rounds', type=int, default=22, required=False,
                    help='number of epochs for early stopping')
parser.add_argument('--lr_schedule_round', type=int, default=30, required=False,
                    help='number of epochs for learning rate scheduling')

parser.add_argument('--train_batch_size', type=int, default=8, required=False,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=8, required=False,
                    help='eval batch size')

parser.add_argument('--model_save_path', type=str, default='models/model_hindi_sentiment_mlm/', required=False,
                    help='seed')

#parser.add_argument('--model_save_path', type=str, default='../models/model_hindi_sentiment/', required=False,
#                    help='seed')

parser.add_argument('--wandb_logging', type=bool, default=True, required=False,
                    help='wandb logging needed')

parser.add_argument('--seed', type=int, default=42, required=False,
                    help='seed')

parser.add_argument("--ismlm", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Run on MLM training mode or task inference mode")


args, _ = parser.parse_known_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

##Hindi dataset concatenation
df1 = pd.read_csv("../data/hindi_sentiment/IIITH_Codemixed.txt", sep='\t',usecols=[1,2])
df2 = pd.read_csv("../data/MSH-Comics-Sarcasm/hindi_sarcasm.txt", sep='\t')
df2 = df2.fillna("")
df3 = pd.read_csv("../data/MSH-Comics-Sarcasm/hindi_humour-codemix.txt", sep='\t')

#df = pd.read_csv(args.train_data, sep='\t',usecols=[1,2])

df2.columns = df1.columns
df3.columns = df1.columns

#df = pd.DataFrame()
df = pd.concat([df1, df2],axis=0, ignore_index=True)
df.columns = ['text','category']
kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
for train_index, test_index in kf.split(df.text):
    break

train_df = df.iloc[train_index]
kf2 = KFold(n_splits=2, shuffle=True, random_state=args.seed)
new_df = df.iloc[test_index]
for val_index, test_index in kf2.split(new_df.text):
    break

val_df = new_df.iloc[val_index]
test_df = new_df.iloc[test_index]

#train_df = train_df.append(train_df[train_df['category'] == "Negative"])
#train_df.sample(frac=1)

print (train_df.shape, val_df.shape, test_df.shape)

train_df.head(5)

train_df.text = train_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
val_df.text = val_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
test_df.text = test_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))

train_df.text.apply(lambda x: len(x)).describe()

train_df.text.apply(lambda x: len(x.split())).describe()

model_save_dir = args.model_save_path

try:
    os.makedirs(model_save_dir)
except OSError:
    pass


from torch.utils.data import Dataset
import torch
import random
from sklearn.preprocessing import OneHotEncoder

def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

def collate_mlm(batch, return_clean=False):

    input_lens = [len(x[0]) for x in batch]
    max_x_len = max(input_lens)

    # chars
    chars_pad = [tf.keras.preprocessing.sequence.pad_sequences([x[0]], maxlen=args.max_text_len)[0].tolist() for x in batch]
    chars = np.stack(chars_pad)
    
    subwords = [x[1] for x in batch]
    for i, subword in enumerate(subwords):
        if len(subword) < args.max_text_len:
            subwords[i] = subwords[i] + [[0]*args.max_word_char_len]
    subwords = [[subword] for subword in subwords]
    
    subwords = np.concatenate(subwords,0)
    
    # labels
    if not return_clean:
      labels_pad = [tf.keras.preprocessing.sequence.pad_sequences([x[2]], maxlen=args.max_text_len)[0].tolist() for x in batch]
      labels = np.stack(labels_pad)
    else:
      labels = np.stack([x[2] for x in batch])


    # masks
    masks = [tf.keras.preprocessing.sequence.pad_sequences([x[3]], maxlen=args.max_text_len)[0].tolist() for x in batch]
    masks = np.stack(masks)

    # position
    position = [pad1d(range(1, len + 1), max_x_len) for len in input_lens]
    position = np.stack(position)
    
    chars = torch.tensor(chars).long()
    try:
        subwords = torch.tensor(subwords).long()
    except:
        print (subwords)
    labels = torch.tensor(labels).long()
    position = torch.tensor(position).long()
    masks = torch.tensor(masks).long()

    output = {"mlm_input": chars,
              "mlm_subword_input": subwords,
              "mlm_label": labels,
              "input_position": position,
              "masks": masks}

    return output

class HITDataset(Dataset):
    def __init__(self, texts, word_tokenizer, char_tokenizer, sos_index, eos_index, mask_index, pad_index, \
                 sos_index_char, eos_index_char, mask_index_char, pad_index_char, labels=None, return_clean=False):
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.texts = texts
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.mask_index = mask_index
        self.pad_index = pad_index
        self.sos_index_char = sos_index_char
        self.eos_index_char = eos_index_char
        self.mask_index_char = mask_index_char
        self.pad_index_char = pad_index_char
        self.return_clean = return_clean
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        t = self.texts[item]
        if not self.return_clean:
          t1_random, char_inputs, t1_label, masks = self.random_word(t, None)
        else:
          label = self.labels[item]
          t1_random, char_inputs, t1_label, masks = self.random_word(t, label)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        mlm_input = t1_random[0] #[self.sos_index] + t1_random[0] + [self.eos_index] #  3，1，2
        mlm_label = t1_label #[self.pad_index] + t1_label + [self.pad_index]
        char_inputs = char_inputs[0] #[[[self.sos_index_char]*args.max_word_char_len] + char_inputs[0] + [[self.eos_index_char]*args.max_word_char_len]]
        
        return mlm_input, char_inputs, mlm_label, masks[0]

    def random_word(self, sentence, label):
        output_label = []
        word_train_inputs = self.word_tokenizer.texts_to_sequences([sentence])
        word_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs, maxlen=args.max_text_len, padding='post')
        #print (word_train_inputs)
        
        sentence_ = " ".join([str(self.word_tokenizer.index_word[i]) for i in word_train_inputs[0]])
        subword_train_inputs = np.asarray(data.data_utils.subword_tokenization(sentence_, self.char_tokenizer, args.max_text_len, args.max_word_char_len, True))[np.newaxis,:].tolist()
        
        masks = np.zeros_like(np.asarray(word_train_inputs))

        if self.return_clean == True:
          #enc = OneHotEncoder()
          #labels = enc.fit_transform(np.array(self.labels).reshape(-1,1))
          #labels = data.data_utils.compute_output_arrays(val_df, 'target')[:,np.newaxis]
          return word_train_inputs, subword_train_inputs, label, masks

        for i, char in enumerate(word_train_inputs[0]):
            prob = random.random()
            if char != 0 and prob < 0.15:
                masks[0,i] = 1
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    word_train_inputs[0][i] = self.mask_index
                    subword_train_inputs[0][i] = [self.mask_index_char]*args.max_word_char_len
                # 10% randomly change token to random token
                elif prob < 0.9:
                    word_train_inputs[0][i] = random.randrange(len(self.word_tokenizer.index_word)-4)
                    try:
                        subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                    except:
                        print (i, word_train_inputs[0][i], len(subword_train_inputs[0]))
                        subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                    subword_train_inputs[0][i] = self.char_tokenizer.texts_to_sequences([subword_train_inputs[0][i]])
                    subword_train_inputs[0][i] = tf.keras.preprocessing.sequence.pad_sequences(subword_train_inputs[0][i], maxlen=args.max_word_char_len, padding='post')[0].tolist()
                # 10% randomly change token to current token
                else:
                    word_train_inputs[0][i] = char
                    try:
                        subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                    except:
                        print (i, word_train_inputs[0][i], len(subword_train_inputs[0]))
                        subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                    subword_train_inputs[0][i] = self.char_tokenizer.texts_to_sequences([subword_train_inputs[0][i]])
                    subword_train_inputs[0][i] = tf.keras.preprocessing.sequence.pad_sequences(subword_train_inputs[0][i], maxlen=args.max_word_char_len, padding='post')[0].tolist()
                    
                output_label.append(char)

            else:
                word_train_inputs[0][i] = char
                try:
                    subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                except:
                    print (i, word_train_inputs[0][i], len(subword_train_inputs[0]))
                    subword_train_inputs[0][i] = str(self.word_tokenizer.index_word[word_train_inputs[0][i]])
                subword_train_inputs[0][i] = self.char_tokenizer.texts_to_sequences([subword_train_inputs[0][i]])
                subword_train_inputs[0][i] = tf.keras.preprocessing.sequence.pad_sequences(subword_train_inputs[0][i], maxlen=args.max_word_char_len, padding='post')[0].tolist()

                output_label.append(char)
        
        #print (np.array(word_train_inputs).shape, np.array(subword_train_inputs).shape,np.array(output_label).shape)
        
        return word_train_inputs, subword_train_inputs, output_label, masks



data.custom_tokenizers.custom_wp_tokenizer(train_df.text.values, args.model_save_path, args.model_save_path)
tokenizer = BertTokenizer.from_pretrained(args.model_save_path)

word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1)
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1)

word_tokenizer.fit_on_texts(train_df.text.values)
char_tokenizer.fit_on_texts(train_df.text.values)

word_tokenizer.word_index['SOS'] = len(word_tokenizer.word_index)
word_tokenizer.index_word[len(word_tokenizer.word_index)] = 'SOS'
word_tokenizer.word_index['EOS'] = len(word_tokenizer.word_index)
word_tokenizer.index_word[len(word_tokenizer.word_index)] = 'EOS'
word_tokenizer.word_index['MASK'] = len(word_tokenizer.word_index)
word_tokenizer.index_word[len(word_tokenizer.word_index)] = 'MASK'
word_tokenizer.word_index['PAD'] = 0
word_tokenizer.index_word[0] = 'PAD'

char_tokenizer.word_index['SOS'] = len(char_tokenizer.word_index)
char_tokenizer.index_word[len(char_tokenizer.word_index)] = 'SOS'
char_tokenizer.word_index['EOS'] = len(char_tokenizer.word_index)
char_tokenizer.index_word[len(char_tokenizer.word_index)] = 'EOS'
char_tokenizer.word_index['MASK'] = len(char_tokenizer.word_index)
char_tokenizer.index_word[len(char_tokenizer.word_index)] = 'MASK'
char_tokenizer.word_index['PAD'] = 0
char_tokenizer.index_word[0] = 'PAD'

sentence = train_df.text.iloc[0]
outputs = [1]*len(train_df)

dataset = HITDataset(train_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                      word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                     char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                     outputs, True)
'''
dataset = HITDataset(train_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                      word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                     char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                     )
'''
word_train_inputs, subword_train_inputs, output_label, masks = dataset.random_word(sentence, [1])


train_dataset = HITDataset(train_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                      word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                     char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                     )
val_dataset = HITDataset(val_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                      word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                     char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                     )


from torch.utils.data import DataLoader


train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch, True),num_workers=1, shuffle=True)
valid_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch, True), num_workers=1, shuffle=True)
'''
train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch),num_workers=1, shuffle=True)
valid_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch), num_workers=1, shuffle=True)
'''

val_dataset_clean = HITDataset(val_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                      word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                     char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, True)
valid_data_loader_clean = DataLoader(val_dataset_clean, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch, True), num_workers=1, shuffle=False)


import torch
import torch.nn as nn
from src.models.torch_layers import Encoder, TimeDistributed, EncoderWithoutEmbedding, Embedder, PositionalEncoding, PositionalEncoder

class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, char_vocab, word_vocab, d_model, max_len ,N, heads, outer_attention=True, dropout=.1, seq_output=False):
        super().__init__()
        self.seq_output = seq_output
        self.d_model = d_model
        self.word_embed = Embedder(word_vocab, d_model)
        self.char_encoder = TimeDistributed(Encoder(char_vocab, d_model, max_len, N, heads, outer_attention))
        self.word_encoder = EncoderWithoutEmbedding(d_model, max_len, N, heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, word_src, char_src_mask, word_src_mask):
        char_outputs = self.char_encoder(src, char_src_mask)
        char_outputs = char_outputs.view(src.shape[0],src.shape[1],src.shape[2],self.d_model)
        char_outputs = char_outputs.mean(axis=2)
        
        if word_src is not None:
            char_outputs = char_outputs + self.word_embed(word_src)

        e_outputs = self.word_encoder(char_outputs, word_src_mask)
        
        e_outputs = self.dropout(e_outputs)

        return e_outputs


class HITLM(nn.Module):
    """
    Masked Language Model
    """

    def __init__(self, backbone, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.backbone = backbone
        self.mask_lm = MaskedLanguageModel(self.backbone.d_model, vocab_size)

    def forward(self, src, word_src, char_src_mask, word_src_mask):
        x = self.backbone(src, word_src, char_src_mask, word_src_mask)
        return self.mask_lm(x)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class HITPM(nn.Module):
    """
    Masked Language Model
    """

    def __init__(self, backbone):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.backbone = backbone
        self.mask_lm = PredictionLanguageModel(self.backbone.d_model)

    def forward(self, src, word_src, char_src_mask, word_src_mask, vocab_size):
        x = self.backbone(src, word_src, char_src_mask, word_src_mask)
        return self.mask_lm(x, vocab_size)


class PredictionLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        #self.linear = nn.Linear(hidden, 100)
        #self.avgpool = nn.AvgPool2d((20, 1))
        ##dropout
        self.linear1 = nn.Linear(hidden, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear31 = nn.Linear(256, 1)
        self.linear32 = nn.Linear(256, 2)
        self.linear33 = nn.Linear(256, 3)
        ##Linear
        self.softmax = nn.Softmax()

    def forward(self, x, vocab_size):
        x = torch.mean(x, dim=1)#self.avgpool(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = nn.Dropout(0.4)(x)
        if vocab_size == 1:
          x = self.linear31(x)
        elif vocab_size == 2:
          x = self.linear32(x)
        else:
          x = self.linear33(x)
        #return nn.functional.relu(x)
        return nn.functional.softmax(x, dim=1)

#print(args.ismlm)

if args.ismlm == True:
    #MLM pre-training step
    
    backbone = HierarchicalTransformerEncoder(char_vocab=len(char_tokenizer.word_index.keys()),\
                                      word_vocab=len(word_tokenizer.word_index.keys()),\
                                      d_model=args.emb_dim, \
                                      max_len=args.max_text_len, \
                                      N=4, heads=8)

    lm_model = HITLM(backbone,len(word_tokenizer.word_index))
    #lm_model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'cor_master_model.pth')))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lm_model.parameters(), lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    lm_model.to(device)
    args.epochs = 100

    best_loss = 999
    patience = 0
    for epoch in tqdm(range(args.epochs)):
      if patience < args.early_stopping_rounds:
        lm_model.train()
        train_loss = 0
        for batch in train_data_loader:
            out = lm_model(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                   torch.where(batch['mlm_input'] > 0, 1,0).to(device))
            #out = out# * batch['masks'].unsqueeze(2).to(device)
            out = out# * batch['masks'].unsqueeze(2).to(device)
            optimizer.zero_grad()
            #loss = loss_fn(out, batch["mlm_label"].to(device))
            loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_data_loader)
        
        val_loss = 0
        acc = 0
        with torch.no_grad():
            for batch in valid_data_loader:
                out = lm_model(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                       torch.where(batch['mlm_input'] > 0, 1,0).to(device))
                #out = out# * batch['masks'].unsqueeze(2).to(device)
                out = out# * batch['masks'].unsqueeze(2).to(device)
                optimizer.zero_grad()
                #loss = loss_fn(out, batch["mlm_label"].to(device))
                loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
                val_loss += loss.item()
                acc += accuracy_score(out.argmax(-1).detach().cpu().numpy().flatten(), batch["mlm_label"].detach().cpu().numpy().flatten())

        val_loss /= len(valid_data_loader)
        acc /= len(valid_data_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
          torch.save(lm_model.state_dict(), os.path.join(args.model_save_path, 'cor_master_model.pth'))
          best_loss = val_loss
          patience = 0
        else:
          patience += 1

        print ("Epoch: {}, Train loss: {}, Val loss: {}, Val accuracy: {}".format(epoch, train_loss, val_loss, acc))

else:
    
    df = df2    #Use df1 for sentiment, df2 for sarcasm, df3 for humour
    df.columns = ['text','category']
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for train_index, test_index in kf.split(df.text):
        break

    train_df = df.iloc[train_index]
    kf2 = KFold(n_splits=2, shuffle=True, random_state=args.seed)
    new_df = df.iloc[test_index]
    for val_index, test_index in kf2.split(new_df.text):
        break

    val_df = new_df.iloc[val_index]
    test_df = new_df.iloc[test_index]

    train_df.text = train_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
    val_df.text = val_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
    test_df.text = test_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
    
    enc = LabelEncoder()

    train_outputs = enc.fit_transform(data.data_utils.compute_output_arrays(train_df, 'category')[:,np.newaxis])
    val_outputs = enc.transform(data.data_utils.compute_output_arrays(val_df, 'category')[:,np.newaxis])

    train_dataset = HITDataset(train_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                          word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                         char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                         train_outputs, True)
    val_dataset = HITDataset(val_df.text.values.tolist(), word_tokenizer, char_tokenizer, \
                          word_tokenizer.word_index['SOS'],word_tokenizer.word_index['EOS'],word_tokenizer.word_index['MASK'], 0, \
                         char_tokenizer.word_index['SOS'],char_tokenizer.word_index['EOS'],char_tokenizer.word_index['MASK'], 0, \
                         val_outputs, True)
                         
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch, True),num_workers=1, shuffle=True)
    valid_data_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, collate_fn=lambda batch: collate_mlm(batch, True), num_workers=1, shuffle=True)

    """##Wihtout MLM"""
    
    
    backbone = HierarchicalTransformerEncoder(char_vocab=len(char_tokenizer.word_index.keys()),\
                                      word_vocab=len(word_tokenizer.word_index.keys()),\
                                      d_model=args.emb_dim, \
                                      max_len=args.max_text_len, \
                                      N=4, heads=8)

    lm_model = HITPM(backbone)
    '''
    for param in backbone.parameters():
      param.requires_grad = False
    '''

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lm_model.parameters(), lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    lm_model.to(device)
    args.epochs = 100

    best_loss = 999
    patience = 0
    for epoch in tqdm(range(args.epochs)):
      if patience < args.early_stopping_rounds:
        lm_model.train()
        train_loss = 0
        for batch in train_data_loader:
            out = lm_model(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                   torch.where(batch['mlm_input'] > 0, 1,0).to(device),len(set(train_outputs)))
            out = out# * batch['masks'].unsqueeze(2).to(device)
            #out = out * batch['masks'].unsqueeze(2).to(device)
            optimizer.zero_grad()
            loss = loss_fn(out, batch["mlm_label"].to(device))
            #loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_data_loader)
        
        val_loss = 0
        acc = 0
        with torch.no_grad():
            for batch in valid_data_loader:
                out = lm_model(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                       torch.where(batch['mlm_input'] > 0, 1,0).to(device),len(set(val_outputs)))
                out = out# * batch['masks'].unsqueeze(2).to(device)
                #out = out * batch['masks'].unsqueeze(2).to(device)
                optimizer.zero_grad()
                loss = loss_fn(out, batch["mlm_label"].to(device))
                #loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
                val_loss += loss.item()
                acc += accuracy_score(out.argmax(-1).detach().cpu().numpy().flatten(), batch["mlm_label"].detach().cpu().numpy().flatten())

        val_loss /= len(valid_data_loader)
        acc /= len(valid_data_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
          torch.save(lm_model.state_dict(), os.path.join(args.model_save_path, 'noob_sarcasm_model.pth'))
          best_loss = val_loss
          patience = 0
        else:
          patience += 1

        print ("Epoch: {}, Train loss: {}, Val loss: {}, Val accuracy: {}".format(epoch, train_loss, val_loss, acc))



    """##With pretraining MLM"""

    backbone = HierarchicalTransformerEncoder(char_vocab=len(char_tokenizer.word_index.keys()),\
                                      word_vocab=len(word_tokenizer.word_index.keys()),\
                                      d_model=args.emb_dim, \
                                      max_len=args.max_text_len, \
                                      N=4, heads=8)

    pre_model = HITLM(backbone,len(word_tokenizer.word_index))

    pre_model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'cor_master_model.pth')))

    '''
    for param in pre_model.backbone.parameters():
      param.requires_grad = False
    '''

    lm_model_2 = HITPM(pre_model.backbone)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lm_model_2.parameters(), lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    lm_model_2.to(device)
    args.epochs = 100

    best_loss = 999
    patience = 0
    for epoch in tqdm(range(args.epochs)):
      if patience < args.early_stopping_rounds:
        lm_model_2.train()
        train_loss = 0
        for batch in train_data_loader:
            out = lm_model_2(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                   torch.where(batch['mlm_input'] > 0, 1,0).to(device),len(set(train_outputs)))
            out = out# * batch['masks'].unsqueeze(2).to(device)
            #out = out * batch['masks'].unsqueeze(2).to(device)
            optimizer.zero_grad()
            loss = loss_fn(out, batch["mlm_label"].to(device))
            #loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_data_loader)
        
        val_loss = 0
        acc = 0
        with torch.no_grad():
            for batch in valid_data_loader:
                out = lm_model_2(batch['mlm_subword_input'].to(device),batch['mlm_input'].to(device), torch.where(batch['mlm_subword_input'] > 0, 1,0).to(device), \
                       torch.where(batch['mlm_input'] > 0, 1,0).to(device), len(set(val_outputs)))
                out = out# * batch['masks'].unsqueeze(2).to(device)
                #out = out * batch['masks'].unsqueeze(2).to(device)
                optimizer.zero_grad()
                loss = loss_fn(out, batch["mlm_label"].to(device))
                #loss = loss_fn(out.transpose(1, 2), batch["mlm_label"].to(device))
                val_loss += loss.item()
                acc += accuracy_score(out.argmax(-1).detach().cpu().numpy().flatten(), batch["mlm_label"].detach().cpu().numpy().flatten())

        val_loss /= len(valid_data_loader)
        acc /= len(valid_data_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
          torch.save(lm_model_2.state_dict(), os.path.join(args.model_save_path, 'mast_2_model.pth'))
          print("Model saved")
          best_loss = val_loss
          patience = 0
        else:
          patience += 1

        print ("Epoch: {}, Train loss: {}, Val loss: {}, Val accuracy: {}".format(epoch, train_loss, val_loss, acc))
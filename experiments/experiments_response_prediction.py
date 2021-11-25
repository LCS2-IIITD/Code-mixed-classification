from __future__ import absolute_import

import os
import sys

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

#parser.add_argument('--train_data', type=str, default='./drive/My Drive/CMC/data/IIITH_Codemixed.txt', required=False,
#                    help='train data')
parser.add_argument('--data_path', type=str, default='data/IITMadras-CodeMixResponse/hindi/', required=False,
                    help='data')

parser.add_argument('--max_text_len', type=int, default=256, required=False,
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

parser.add_argument('--epochs', type=int, default=10, required=False,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=.001, required=False,
                    help='learning rate')
parser.add_argument('--early_stopping_rounds', type=int, default=10, required=False,
                    help='number of epochs for early stopping')
parser.add_argument('--lr_schedule_round', type=int, default=30, required=False,
                    help='number of epochs for learning rate scheduling')

parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                    help='eval batch size')

#parser.add_argument('--model_save_path', type=str, default='./drive/My Drive/CMC/models/model_hindi_sentiment/', required=False,
#                    help='seed')

parser.add_argument('--model_save_path', type=str, default='models/hindi_Response/', required=False,
                    help='seed')

args, _ = parser.parse_known_args()

BATCH_SIZE = args.train_batch_size
MAX_LENGTH = args.max_text_len
num_layers = args.n_layers
d_model = args.n_units
dff = args.emb_dim
num_heads = 8
checkpoint_path = args.model_save_path
EPOCHS = args.epochs
data_path = args.data_path

with open(os.path.join(data_path,'p-dialog-dstc2-train.json'),'r') as fp:
  train=json.load(fp)

with open(os.path.join(data_path,'p-dialog-dstc2-test.json'),'r') as fp:
  test=json.load(fp)

#src = pd.read_csv(os.path.join(data_path,'train.src'),header=None,sep='\t')
#target = pd.read_csv(os.path.join(data_path,'train_cm.tgt'),header=None,sep='\t')
#train = np.array(train)
#test = np.array(test)

train_df = pd.DataFrame()
train_df['source'] = [" ".join(x) for x in train[0]]
train_df['target'] = [(" ".join(x)).replace('<GO>','[CLS]').strip() for x in train[1]]
train_df['target_output'] = [" ".join(x) for x in train[2]]
train_df = train_df.iloc[:-2001]
train_df['target_input'] = train_df['target']
#train_df['target_output'] = train[:,2]
train_df['target'] = train_df.target.apply(lambda x: "{} [EOS]".format(x))

#src = pd.read_csv(os.path.join(data_path,'dev.src'),header=None,sep='\t')
#target = pd.read_csv(os.path.join(data_path,'dev_cm.tgt'),header=None,sep='\t')

val_df = pd.DataFrame()
val_df['source'] = [" ".join(x) for x in train[0]]
val_df['target'] = [(" ".join(x)).replace('<GO>','[CLS]').strip() for x in train[1]]
val_df['target_output'] = [" ".join(x) for x in train[2]]
val_df = val_df.iloc[-2001:]
val_df['target_input'] = val_df['target']
#val_df['target_output'] = train[:,2]
val_df['target'] = val_df.target.apply(lambda x: "{} [EOS]".format(x))
#val_df['target_input'] = val_df.target.apply(lambda x: "[CLS] {}".format(x))
#val_df['target_output'] = val_df.target.apply(lambda x: "{} [SEP]".format(x))
#val_df['target'] = val_df.target.apply(lambda x: "[CLS] {} [SEP]".format(x))

#src = pd.read_csv(os.path.join(data_path,'test.src'),header=None,sep='\t')
#target = pd.read_csv(os.path.join(data_path,'test_cm.tgt'),header=None,sep='\t')

test_df = pd.DataFrame()
test_df['source'] = [" ".join(x) for x in test[0]]
test_df['target'] = [(" ".join(x)).replace('<GO>','[CLS]').strip() for x in test[1]]
test_df['target_output'] = [" ".join(x) for x in test[2]]
test_df['target_input'] = test_df['target']
#test_df['target_output'] = test_df.target.apply(lambda x: "{} [SEP]".format(x))
test_df['target'] = test_df.target.apply(lambda x: "{} [EOS]".format(x))

#examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
#                               as_supervised=True)
#train_examples, val_examples = examples['train'], examples['validation']



def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()

tot_train = train_df
tot_train = tot_train.append(val_df)

print(len(tot_train), len(test_df))

"""##Let's prepare dataset to run on HIT model"""

from tqdm import tqdm

import tokenizers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig, BertTokenizer

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src import data, models
from src.models.models import *


data.custom_tokenizers.custom_wp_tokenizer(train_df.source.values, args.model_save_path, args.model_save_path)
tokenizer = BertTokenizer.from_pretrained(args.model_save_path)

word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1)
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1)

word_tokenizer.fit_on_texts(train_df.source.values)
char_tokenizer.fit_on_texts(train_df.source.values)

word_tokenizer.fit_on_texts(train_df.target.values)
char_tokenizer.fit_on_texts(train_df.target.values)

##

transformer_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(train_df, 'source', tokenizer, args.max_char_len)
#transformer_train_inputs = []

word_train_inputs = word_tokenizer.texts_to_sequences(train_df.source.values)
word_train_inputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in word_train_inputs]
word_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs, maxlen=args.max_text_len, padding='post')
#word_tokenizer.sequences_to_texts(word_train_inputs)
subword_train_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len)\
                         for text in tqdm(train_df.source.values)])

char_train_inputs = char_tokenizer.texts_to_sequences(train_df.source.values)
char_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_train_inputs, maxlen=args.max_char_len, padding='post')

train_outputs = data.data_utils.compute_output_arrays(train_df, 'target')

transformer_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(val_df, 'source', tokenizer, args.max_char_len)
#transformer_val_inputs = []

word_val_inputs = word_tokenizer.texts_to_sequences(val_df.source.values)
word_val_inputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in word_val_inputs]
word_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_val_inputs, maxlen=args.max_text_len, padding='post')

subword_val_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len)\
                         for text in tqdm(val_df.source.values)])

char_val_inputs = char_tokenizer.texts_to_sequences(val_df.source.values)
char_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_val_inputs, maxlen=args.max_char_len, padding='post')

val_outputs = data.data_utils.compute_output_arrays(val_df, 'target')

transformer_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(test_df, 'source', tokenizer, args.max_char_len)
#transformer_test_inputs = []

word_test_inputs = word_tokenizer.texts_to_sequences(test_df.source.values)
word_test_inputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in word_test_inputs]
word_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs, maxlen=args.max_text_len, padding='post')

subword_test_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len)
                         for text in tqdm(test_df.source.values)])

char_test_inputs = char_tokenizer.texts_to_sequences(test_df.source.values)
char_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_test_inputs, maxlen=args.max_char_len, padding='post')

test_outputs = data.data_utils.compute_output_arrays(test_df, 'target')

train_outputs = word_tokenizer.texts_to_sequences(train_df.target.values)
train_outputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in train_outputs]
train_outputs = tf.keras.preprocessing.sequence.pad_sequences(train_outputs, maxlen=args.max_text_len, padding='post')

val_outputs = word_tokenizer.texts_to_sequences(val_df.target.values)
val_outputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in val_outputs]
val_outputs = tf.keras.preprocessing.sequence.pad_sequences(val_outputs, maxlen=args.max_text_len, padding='post')

test_outputs = word_tokenizer.texts_to_sequences(test_df.target.values)
test_outputs = [[len(word_tokenizer.word_index)] + x + [len(word_tokenizer.word_index)+1] for x in test_outputs]
test_outputs = tf.keras.preprocessing.sequence.pad_sequences(test_outputs, maxlen=args.max_text_len, padding='post')


#train_outputs = tf.keras.utils.to_categorical(train_outputs,num_classes=len(word_tokenizer.word_index)+1)
#val_outputs = tf.keras.utils.to_categorical(val_outputs, num_classes=len(word_tokenizer.word_index)+1)
#test_outputs = tf.keras.utils.to_categorical(test_outputs,num_classes=len(word_tokenizer.word_index)+1)


'''
tfidf1 = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_df=.6,min_df=2)
tfidf2 = TfidfVectorizer(analyzer='char_wb',ngram_range=(1,3), max_df=.6,min_df=2)

tfidf1.fit(train_df.source)
tfidf2.fit(train_df.source)

train_tfidf = np.hstack([tfidf1.transform(train_df.source).toarray(),tfidf2.transform(train_df.source).toarray()])
val_tfidf = np.hstack([tfidf1.transform(val_df.source).toarray(),tfidf2.transform(val_df.source).toarray()])
test_tfidf = np.hstack([tfidf1.transform(test_df.source).toarray(),tfidf2.transform(test_df.source).toarray()])


print (transformer_train_inputs.shape, subword_train_inputs.shape, word_train_inputs.shape, char_train_inputs.shape, train_tfidf.shape, train_outputs.shape)
print (transformer_val_inputs.shape, subword_val_inputs.shape, word_val_inputs.shape, char_val_inputs.shape, val_tfidf.shape, val_outputs.shape)
print (transformer_test_inputs.shape, subword_test_inputs.shape, word_test_inputs.shape, char_test_inputs.shape, test_tfidf.shape, test_outputs.shape)
'''

#### Modeling

n_words = len(word_tokenizer.word_index)+2
n_chars = len(char_tokenizer.word_index)+1
n_subwords = tokenizer.vocab_size
#tfidf_shape = train_tfidf.shape[1]
n_out = len(word_tokenizer.word_index)+2#args.max_text_len

char_test_inputs.shape


from tqdm import tqdm

'''
all_preds = []
test_df = test_df.iloc[:800]

for i in tqdm(range(test_df.shape[0])):
  all_preds.append(translate(test_df.source.iloc[i]))
'''

#test_df['predicted'] = seq_tests

#!pip install rouge_score

from rouge_score import rouge_scorer
import nltk.translate.bleu_score as bleu

def bleu_score(text1,text2):
    return bleu.sentence_bleu([text1.lower().split()],text2.lower().split())

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from itertools import chain, product


def _generate_enums(hypothesis, reference, preprocess=str.lower):
    """
    Takes in string inputs for hypothesis and reference and returns
    enumerated word lists for each of them

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :preprocess: preprocessing method (default str.lower)
    :type preprocess: method
    :return: enumerated words list
    :rtype: list of 2D tuples, list of 2D tuples
    """
    hypothesis_list = list(enumerate(preprocess(hypothesis).split()))
    reference_list = list(enumerate(preprocess(reference).split()))
    return hypothesis_list, reference_list


def exact_match(hypothesis, reference):
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(hypothesis_list, reference_list)



def _match_enums(enum_hypothesis_list, enum_reference_list):
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.

    :param enum_hypothesis_list: enumerated hypothesis list
    :type enum_hypothesis_list: list of tuples
    :param enum_reference_list: enumerated reference list
    :type enum_reference_list: list of 2D tuples
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()
):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list:
    :type enum_hypothesis_list:
    :param enum_reference_list:
    :type enum_reference_list:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    stemmed_enum_list1 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list
    ]

    stemmed_enum_list2 = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = _match_enums(
        stemmed_enum_list1, stemmed_enum_list2
    )

    enum_unmat_hypo_list = (
        list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []
    )

    enum_unmat_ref_list = (
        list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []
    )

    enum_hypothesis_list = list(
        filter(lambda x: x[0] not in enum_unmat_hypo_list, enum_hypothesis_list)
    )

    enum_reference_list = list(
        filter(lambda x: x[0] not in enum_unmat_ref_list, enum_reference_list)
    )

    return word_match, enum_hypothesis_list, enum_reference_list


def stem_match(hypothesis, reference, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis:
    :type hypothesis:
    :param reference:
    :type reference:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that
                   implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)



def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype:  list of tuples, list of tuples, list of tuples

    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(
            chain(
                *[
                    [
                        lemma.name()
                        for lemma in synset.lemmas()
                        if lemma.name().find("_") < 0
                    ]
                    for synset in wordnet.synsets(enum_hypothesis_list[i][1])
                ]
            )
        ).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append(
                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])
                )
                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of mapped tuples
    :rtype: list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )



def _enum_allign_words(
    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer(), wordnet=wordnet
):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )

    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )


def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_allign_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )



def _count_chunks(matches):
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to caluclate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of allign_words)
    :return: Number of chunks a sentence is divided into post allignment
    :rtype: int
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
    reference,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.9,
    beta=3,
    gamma=0.5,
):
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score('this is a cat', 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean



def meteor_score(
    references,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    wordnet=wordnet,
    alpha=0.9,
    beta=3,
    gamma=0.5,
):
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis

    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    >>> reference3 = 'It is the practical guide for the army always to heed the directions of the party'

    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score(['this is a cat'], 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    return max(
        [
            single_meteor_score(
                reference,
                hypothesis,
                stemmer=stemmer,
                wordnet=wordnet,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            for reference in references
        ]
    )

# Creating a reverse dictionary word_tokenizer.sequences_to_texts(word_train_inputs)
reverse_word_map = dict(map(reversed, word_tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices if letter != 0]
    return(words)

# Creating texts 
#list(map(sequence_to_text, word_train_inputs))

from sklearn.metrics import f1_score
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
import nltk
nltk.download('wordnet')

class SeqF1Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, inputs, targets, filename, patience):
    self.model = model
    self.inputs = inputs
    self.targets = targets
    self.filename = filename
    self.patience = patience

    self.best_score = 0
    self.bad_epoch = 0

  def on_epoch_end(self, epoch, logs):
    pred = self.model.predict(self.inputs)
    #print(pred.argmax(-1), self.targets)
    score = flat_f1_score(self.targets, pred.argmax(-1), average='macro')

    if score > self.best_score:
      self.best_score = score
      self.model.save_weights(self.filename)
      print ("\nScore {}. Model saved in {}.".format(score, self.filename))
      self.bad_epoch = 0
    else:
      print ("\nScore {}. Model not saved.".format(score))
      self.bad_epoch += 1

    if self.bad_epoch >= self.patience:
      print ("\nEpoch {}: early stopping.".format(epoch))
      self.model.stop_training = True

"""## Loss and metrics

Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
"""

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(tf.cast(real, tf.dtypes.int64), tf.argmax(pred, -1))
  
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

#import keras.backend as K 
'''
def word_acc(y_true, y_pred):
    a = tf.argmax(y_pred, axis=-1)
    b = tf.math.reduce_max(y_true, axis=-1)
    b = tf.cast(b, K.tf.int64)
    
    total_index = tf.math.count_nonzero(a+b, 1)
    wrong_index = tf.math.count_nonzero(a-b, 1)

    total_index = tf.math.reduce_sum(total_index)
    wrong_index = tf.math.reduce_sum(wrong_index)
    
    correct_index = tf.cast(total_index-wrong_index, dtype=tf.float32)
    total_index = tf.cast(total_index, dtype=tf.float32)
    
    acc = tf.math.divide(correct_index, total_index)
    
    return tf.cast(acc,dtype=tf.float32)
'''

def f1_keras(y_true, y_pred):
    y_pred = tf.argmax(y_pred, -1)
    y_true = tf.cast(y_true, tf.dtypes.int64)
    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



#all_models = {HIT.__name__: HIT, HIT_without_words.__name__:HIT_without_words}#CS_ELMO_without_words.__name__: CS_ELMO_without_words, HAN.__name__: HAN}
#HIT_without_words.__name__:HIT_without_words, CS_ELMO_without_words.__name__: CS_ELMO_without_words, AttentionAt      
#all_models = {HIT_outer.__name__:HIT_outer,HIT.__name__: HIT, HIT_without_words.__name__:HIT_without_words, }

all_models = {HIT_outer.__name__:HIT_outer}

_has_wandb = False


if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

model = None

for model_name, model_ in all_models.items():
    
    for loss in ['ce', 'focal']:
        
        for use_features in [False]:
            model = None
            if use_features == False:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=n_out,max_word_char_len=args.max_word_char_len,\
                                                             seq_output=True,max_text_len=args.max_text_len, max_char_len=args.max_char_len, n_layers=args.n_layers, \
                                                             n_units=args.n_units, emb_dim=args.emb_dim)
            else:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords,n_out=n_out, seq_output=True,\
                    vectorizer_shape=tfidf_shape, max_word_char_len=args.max_word_char_len,max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                                                 n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
            
            if use_features == True:
                print ("Running {} with features for {} loss".format(model_name, loss))
            else:
                print ("Running {} without features for {} loss".format(model_name, loss))

            print (model.summary())

            if loss == 'focal':
                model.compile(loss=loss_function, optimizer='adam', metrics=[accuracy_function, f1_keras]) #binary_crossentropy
            elif loss == 'ce':
                model.compile(loss=loss_function, optimizer='adam', metrics=[accuracy_function, f1_keras]) 

            lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7,\
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
                model_save_path = os.path.join(args.model_save_path, '{}_{}_without_boo_features.h5'.format(model_name, config['loss']))

            

            K.clear_session()

            if use_features == True:
              f1callback = models.utils.SeqF1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf],\
                                                  val_outputs,filename=model_save_path,patience=args.early_stopping_rounds)
              if _has_wandb and args.wandb_logging:
                  wandb.init(project='hindi_sarcasm',config=config)
                  model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs,\
                                        validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs),\
                                                                    epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback, WandbCallback()], verbose=2)
              else:
                  model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs,\
                                        validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs),\
                                                                    epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback], verbose=1)
            else:
                f1callback = SeqF1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs],\
                                                  val_outputs,filename=model_save_path,patience=args.early_stopping_rounds)
                if _has_wandb and args.wandb_logging:
                  wandb.init(project='hindi_sarcasm',config=config)
                  model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs], train_outputs,\
                                        validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs], val_outputs),\
                                                                    epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback, WandbCallback()], verbose=2)
                else:
                  model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs], train_outputs,\
                                        validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs], val_outputs),\
                                                                    epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback], verbose=1)

            model.load_weights(model_save_path)

            
            if use_features == True:
              test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, test_tfidf])
            else:
              test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs])

            
            report = flat_classification_report([i for i in test_outputs],\
                                                        [i for i in test_pred.argmax(-1)])

            f1 = flat_f1_score([i for i in test_outputs],\
                                                        [i for i in test_pred.argmax(-1)], average='weighted')

            new_test_preds = []

            for pred in test_pred.argmax(-1):
              out = []
              for pre in pred:
                if pre == len(word_tokenizer.word_index):
                  continue
                elif pre == len(word_tokenizer.word_index)+1:
                  break
                else:
                  out.append(pre)
              new_test_preds.append(out)

            seq_tests = list(map(sequence_to_text, new_test_preds))
            seq_tests = [" ".join(x) for x in seq_tests]

            test_df['predicted'] = seq_tests
            
            test_df.target = test_df.target.apply(lambda x: x.replace('[CLS]','').strip())
            test_df.predicted = test_df.predicted.apply(lambda x: x.replace('cls','').strip())
            test_df.target = test_df.target.apply(lambda x: x.replace('[SEP]','').strip())
            test_df.predicted = test_df.predicted.apply(lambda x: x.replace('sep','').strip())

            test_df['bleu'] = test_df.apply(lambda x: bleu_score(x.target,x.predicted), axis=1)
            test_df['rouge1'] = test_df.apply(lambda x: scorer.score(x.target,x.predicted)['rouge1'].fmeasure, axis=1)
            test_df['rouge2'] = test_df.apply(lambda x: scorer.score(x.target,x.predicted)['rouge2'].fmeasure, axis=1)
            test_df['rougeL'] = test_df.apply(lambda x: scorer.score(x.target,x.predicted)['rougeL'].fmeasure, axis=1)
            test_df['meteor'] = test_df.apply(lambda x: meteor_score(x.target,x.predicted), axis=1)

            print(test_df[['bleu','rouge1','rouge2', 'rougeL','meteor']].describe())

            test_df.head()

            #test_df.to_csv(os.path.join(args.model_save_path,'full_test.csv'),index=False,sep='\t')

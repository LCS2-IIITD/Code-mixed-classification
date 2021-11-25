import tensorflow as tf
import numpy as np
from .layers import *

def WLSTM(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CLSTM(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_char_len)(char_inputs)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model


def CMSA(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, kernel_size=3, \
         max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_char_len)(char_inputs)
    
    emb = tf.keras.layers.Conv1D(filters=n_units,kernel_size=kernel_size,strides=1,padding='valid',
                            activation='relu')(emb)
    emb = tf.keras.layers.MaxPooling1D(kernel_size)(emb)
    
    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        attention = AttentionWithContext(name='attention')(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, attention, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm, attention]))
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//2)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//4)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    dense = tf.keras.layers.Dense(n_units//8)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def HAN(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, \
        max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)
    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(char_inputs)
    
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(emb)
    for i in range(n_layers-1):
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)

    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_units))(lstm)
    dense = AttentionWithContext(name='char_attention')(dense)
    
    char_model = tf.keras.models.Model(char_inputs, dense)
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    word_encoder = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    #word_embedding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #word_encoder = word_encoder + word_embedding

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_encoder)
    for i in range(n_layers-1):
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)

    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_units))(lstm)
    
    if seq_output == False:
        dense = AttentionWithContext(name='word_attention')(dense)
    else:
        dense = Attention(n_units,name='word_attention')(dense)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([dense, tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(dense)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CS_ELMO(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, \
            max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim == n_units
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(subword_inputs)
    
    bigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=2,strides=1,padding='same',
                            activation='relu'))(emb)
    
    bigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    bigram_position = bigram_embedding_layer(subword_inputs)
    bigram_position = bigram_emb + bigram_position

    bigram_emb = tf.keras.layers.TimeDistributed(Attention(2*n_units))(tf.keras.layers.Concatenate()([bigram_emb,bigram_position]))
    
    trigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=3,strides=1,padding='same',
                            activation='relu'))(emb)
    
    trigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    trigram_position = trigram_embedding_layer(subword_inputs)
    trigram_position = trigram_emb + trigram_position

    trigram_emb = tf.keras.layers.TimeDistributed(Attention(2*n_units))(tf.keras.layers.Concatenate()([trigram_emb,trigram_position]))
    
    word_emb = tf.keras.layers.TimeDistributed(AttentionWithContext())(tf.keras.layers.Concatenate()([bigram_emb,trigram_emb]))
    word_emb = tf.keras.layers.Dense(n_units)(word_emb)
    word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    word_emb = word_emb + word_embbeding

    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(word_emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm,tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def CS_ELMO_without_words(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None, \
            max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim == n_units
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    emb = tf.keras.layers.Embedding(char_vocab_size, emb_dim, input_length = max_word_char_len)(subword_inputs)
    
    bigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=2,strides=1,padding='same',
                            activation='relu'))(emb)
    
    bigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    bigram_position = bigram_embedding_layer(subword_inputs)
    bigram_position = bigram_emb + bigram_position

    bigram_emb = tf.keras.layers.TimeDistributed(Attention(2*n_units))(tf.keras.layers.Concatenate()([bigram_emb,bigram_position]))
    
    trigram_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=n_units,\
                                                                        kernel_size=3,strides=1,padding='same',
                            activation='relu'))(emb)
    
    trigram_embedding_layer = PositionEmbedding(max_word_char_len, emb_dim) #tf.keras.layers.TimeDistributed(PositionEmbedding(max_word_char_len, emb_dim))
    trigram_position = trigram_embedding_layer(subword_inputs)
    trigram_position = trigram_emb + trigram_position

    trigram_emb = tf.keras.layers.TimeDistributed(Attention(2*n_units))(tf.keras.layers.Concatenate()([trigram_emb,trigram_position]))
    
    word_emb = tf.keras.layers.TimeDistributed(AttentionWithContext())(tf.keras.layers.Concatenate()([bigram_emb,trigram_emb]))
    word_emb = tf.keras.layers.Dense(n_units)(word_emb)
    #word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #word_emb = word_emb + word_embbeding

    if n_layers == 1:
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(word_emb)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
    else:
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(word_emb)
        for i in range(n_layers-2):
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
        if seq_output == False:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=False))(lstm)
        else:
            lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))(lstm)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        dense = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([lstm,tfidf]))
    else:
        dense = tf.keras.layers.Dense(n_units)(lstm)
    dense = tf.keras.layers.Dropout(.2)(dense)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(dense)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def HIT(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    x = embedding_layer(char_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    x = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    position = embedding_layer(word_inputs)
    
    x = x + position
    
    word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    x = x + word_embbeding

    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

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
        x = transformer_blocks[i](x, mask=None)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    x = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    position = embedding_layer(word_inputs)
    
    x = x + position
    
    word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    x = x + word_embbeding

    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units, outer_attention=True))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def Transformer_outer(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    position = embedding_layer(word_inputs)
    
    word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    x = position + word_embbeding

    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units, outer_attention=True))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def HIT_without_words(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    x = embedding_layer(char_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    x = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    position = embedding_layer(word_inputs)
    
    x = x + position
    
    #word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #x = x + word_embbeding

    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def HIT_outer_without_words(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    x = embedding_layer(char_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units, outer_attention=True))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    out = AttentionWithContext(name='char_attention')(x)
    char_model = tf.keras.models.Model(inputs=char_inputs, outputs=out)
    #print (char_model.summary())
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    x = tf.keras.layers.TimeDistributed(char_model)(subword_inputs)
    
    embedding_layer = PositionEmbedding(max_text_len, emb_dim)
    position = embedding_layer(word_inputs)
    
    x = x + position
    
    #word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)(word_inputs)

    #x = x + word_embbeding

    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units, outer_attention=True))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def Transformer(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):

    assert emb_dim%n_heads == 0
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(max_char_len, wpe_vocab_size, emb_dim)
    x = embedding_layer(wpe_inputs)
    
    transformer_blocks = []
    
    for i in range(n_layers):
        transformer_blocks.append(TransformerBlock(emb_dim, n_heads, n_units))
        
    for i in range(n_layers):
        x = transformer_blocks[i](x, mask=None)
    
    if seq_output == False:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

#all_models = {HIT.__name__:HIT, HIT_outer.__name__:HIT_outer, HIT_without_words.__name__: HIT_without_words, HIT_outer_without_words.__name__: HIT_outer_without_words,\
#            Transformer.__name__: Transformer, CS_ELMO.__name__: CS_ELMO, CS_ELMO_without_words.__name__:CS_ELMO_without_words, HAN.__name__: HAN, \
#            CMSA.__name__: CMSA, WLSTM.__name__: WLSTM}


class char_model(tf.keras.Model):
  def __init__(self, max_word_char_len, char_vocab_size, emb_dim, n_heads, n_units, n_layers):
    super(char_model, self).__init__()
    self.TokenAndPositionEmbedding = TokenAndPositionEmbedding(max_word_char_len, char_vocab_size, emb_dim)
    self.n_layers = n_layers
    self.n_units = n_units

    self.transformer_blocks_1 = []
    for i in range(n_layers):
        self.transformer_blocks_1.append(TransformerBlock(emb_dim, n_heads, n_units,outer_attention=True))
    self.attention = AttentionWithContext(name='char_attention')

  def call(self, char_inputs):
    x = self.TokenAndPositionEmbedding(char_inputs)

    for i in range(self.n_layers):
      x = self.transformer_blocks_1[i](x, mask=None)

    x = self.attention(x)

    return x

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.n_units)

class cl_HIT_outer(tf.keras.Model):
  def __init__(self, char_vocab_size, word_vocab_size, n_out, seq_output=False, vectorizer_shape=None, 
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    super(cl_HIT_outer, self).__init__()

    #self.char_inputs = tf.keras.layers.Input((max_word_char_len,), dtype=tf.int32)
    self.vectorizer_shape = vectorizer_shape
    self.seq_output = seq_output
    self.n_layers = n_layers

    self.char_model = char_model(max_word_char_len, char_vocab_size, emb_dim, n_heads, n_units, n_layers)
    self.PositionEmbedding = PositionEmbedding(max_text_len, emb_dim)
    self.word_embbeding = tf.keras.layers.Embedding(word_vocab_size, emb_dim, input_length = max_text_len)    
    self.transformer_blocks_2 = []
    
    for i in range(n_layers):
        #self.transformer_blocks_1.append(TransformerBlock(emb_dim, n_heads, n_units,outer_attention=True))
        self.transformer_blocks_2.append(TransformerBlock(emb_dim, n_heads, n_units,outer_attention=True))
               
    self.GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D()
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense = tf.keras.layers.Dense(n_units)
    #self.dense2 = tf.keras.layers.Dense(n_units)
    self.final_layer = tf.keras.layers.Dense(n_out, activation='softmax')
    self.TimeDistributed = tf.keras.layers.TimeDistributed(self.char_model)

    
  def call(self, inputs):
    
    word_inputs, char_inputs, subword_inputs, transformer_inputs, tfidf = inputs
    #self.char_model.compile(run_eagerly=True)
    y = self.char_model(char_inputs)
    #print(y.shape)

    x = self.TimeDistributed(subword_inputs)

    position = self.PositionEmbedding(word_inputs)
    #print(position.shape)
    
    x = x + position
    
    word_embbeding = self.word_embbeding(word_inputs)

    x = x + word_embbeding

    for i in range(self.n_layers):
      x = self.transformer_blocks_2[i](x, mask=None)
    
    if self.seq_output == False:
        x = self.GlobalAveragePooling1D(x)
    x = self.dropout(x)
    
    if self.vectorizer_shape:
        #tfidf = tf.keras.layers.Input((self.vectorizer_shape,))
        x = self.dense(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = self.dense(x)
    
    learnt_embedding = x
    x = self.dropout(x)
    
    final_output = self.final_layer(x)
    
    return final_output

  def embed(self, inputs):
    
    word_inputs, char_inputs, subword_inputs, transformer_inputs, tfidf = inputs
    #self.char_model.compile(run_eagerly=True)
    y = self.char_model(char_inputs)
    #print(y.shape)

    x = self.TimeDistributed(subword_inputs)

    position = self.PositionEmbedding(word_inputs)
    #print(position.shape)
    
    x = x + position
    
    word_embbeding = self.word_embbeding(word_inputs)

    x = x + word_embbeding

    for i in range(self.n_layers):
      x = self.transformer_blocks_2[i](x, mask=None)
    
    if self.seq_output == False:
        x = self.GlobalAveragePooling1D(x)
    x = self.dropout(x)
    
    if self.vectorizer_shape:
        #tfidf = tf.keras.layers.Input((self.vectorizer_shape,))
        x = self.dense(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = self.dense(x)
    
    learnt_embedding = x
    #x = self.dropout(x)
    
    #final_output = self.final_layer(x)
    
    return learnt_embedding




#-*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import time
import cv2
import keras
#from keras.preprocessing import sequence
import pdb
from keras.layers import Input, Dense, LSTM, Masking,TimeDistributed
from keras.models import Model
from sklearn.utils import shuffle

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_step, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_step = n_lstm_step
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step


    def build_model(self):
        self.encoder = LSTM(self.dim_hidden, return_state=True)
        self.decoder = LSTM(self.dim_hidden, return_sequences=True, return_state=True)
        self.dense = Dense(self.n_words, activation="softmax")
        video_ipt = Input([self.n_video_lstm_step, self.dim_image])
        caption_input = Input((None, self.n_words))
        video = Masking(mask_value=0)(video_ipt)
        image_emb = TimeDistributed(Dense(self.dim_hidden))(video)
        decoder_inputs = Masking(mask_value=0)(caption_input)
        encoder_outputs, state_h, state_c = self.encoder(image_emb)
        encoder_states = [state_h, state_c]

        decoder_outputs, _, _ = self.decoder(decoder_inputs,
                                    initial_state=encoder_states)
        decoder_opt = self.dense(decoder_outputs)
        model = Model([video_ipt, caption_input], decoder_opt)



        encoder_model = Model(video_ipt, encoder_states)

        decoder_state_input_h = Input(shape=(self.dim_hidden,))
        decoder_state_input_c = Input(shape=(self.dim_hidden,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder(
            caption_input, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.dense(decoder_outputs)
        decoder_model = Model(
            [caption_input] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return model, encoder_model, decoder_model



#### global parameters
video_path = './Data/YouTubeClips'
video_feat_path = './Data/Features_VGG'
video_data_path = './Data/video_corpus.csv'
model_path = './ckpt'

#### train parameters
dim_image = 4096
dim_hidden = 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 2000
batch_size = 1
learning_rate = 0.0001

def get_video_data(video_data_path, video_feat_path, train_ratio=1):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    # pdb.set_trace()
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)
    
    shuffle(unique_filenames)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    # pdb.set_trace()
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(wordtoix)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, wordtoix["<bos>"]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idxtoword[sampled_token_index]
        decoded_sentence += sampled_char
        decoded_sentence += " "

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<eos>' or
           len(decoded_sentence) > n_caption_lstm_step):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(wordtoix)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
if __name__ == "__main__":
    train_data, _ = get_video_data(video_data_path, video_feat_path, 1)
    # pdb.set_trace()
    train_captions = train_data['Description'].values

    captions_list = list(train_captions)
    captions = np.array(captions_list, dtype=np.object)

    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)
    idxtoword = dict(zip(wordtoix.values(), wordtoix.keys()))
    # pdb.set_trace()
    # pdb.set_trace()
    np.save('./data/wordtoix', wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save('./data/bias_init_vector', bias_init_vector)

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_step=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    # tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    model_train,encoder_model, decoder_model = model.build_model()
    model_train.load_weights("./model.h5")
    model_train.compile(optimizer=
                        # "sgd",
                        keras.optimizers.Nadam(0.0001), 
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    # pdb.set_tr
    # ace()

# (Pdb) train_data.head()
#            VideoID  Start  End  ...  Language                                    Description                                   video_path
# 64173  m1NR0uNNs5Y    104  110  ...   English  Someone is cutting slices into an onion half.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64174  m1NR0uNNs5Y    104  110  ...   English                      A person slices an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64175  m1NR0uNNs5Y    104  110  ...   English                   A woman is slicing an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64176  m1NR0uNNs5Y    104  110  ...   English                    A chef is slicing an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64177  m1NR0uNNs5Y    104  110  ...   English                    Someone is chopping onions.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy

    start_time = time.time()

    current_video = "./Data/Features_VGG/_0nX-El-ySo_83_93.avi.npy"
    current_feats = np.zeros((1, n_video_lstm_step, dim_image))
    current_feats_val = np.load(current_video)
    ### shape = (80,4096)

    ###下面的意思是读到的numpy不一定有80个timestamp那么长， 但是current_feats必须有80，所以要填充进去 然后记录下来
    ##有数据的特征 mask就是1
    # pdb.set_trace()
    current_feats[0][:len(current_feats_val[0])] = current_feats_val
    print(decode_sequence(current_feats))
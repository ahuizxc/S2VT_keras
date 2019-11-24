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
        video_ipt = Input([self.n_video_lstm_step, self.dim_image])
        caption_input = Input((self.n_caption_lstm_step+1, self.n_words))
        video = Masking(mask_value=0)(video_ipt)
        image_emb = TimeDistributed(Dense(self.dim_hidden))(video)
        decoder_inputs = Masking(mask_value=0)(caption_input)
        encoder_outputs, state_h, state_c = self.encoder(image_emb)
        encoder_states = [state_h, state_c]

        decoder_outputs, _, _ = self.decoder(decoder_inputs,
                                    initial_state=encoder_states)
        decoder_opt = Dense(self.n_words, activation="softmax")(decoder_outputs)
        model = Model([video_ipt, caption_input], decoder_opt)
        return model




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

    model_train = model.build_model()
    model_train.compile(optimizer=
                        # "sgd",
                        keras.optimizers.Nadam(0.0001), 
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []
    # pdb.set_tr
    # ace()

# (Pdb) train_data.head()
#            VideoID  Start  End  ...  Language                                    Description                                   video_path
# 64173  m1NR0uNNs5Y    104  110  ...   English  Someone is cutting slices into an onion half.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64174  m1NR0uNNs5Y    104  110  ...   English                      A person slices an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64175  m1NR0uNNs5Y    104  110  ...   English                   A woman is slicing an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64176  m1NR0uNNs5Y    104  110  ...   English                    A chef is slicing an onion.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy
# 64177  m1NR0uNNs5Y    104  110  ...   English                    Someone is chopping onions.  ./Data/Features_VGG/m1NR0uNNs5Y_104_110.npy

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.loc[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)
        start = 0
        end = 1
        # for start, end in zip(range(0, len(current_train_data), batch_size),range(batch_size, len(current_train_data), batch_size)):
        if 1:
            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
            ### shape = (80,4096)

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))
            ###下面的意思是读到的numpy不一定有80个timestamp那么长， 但是current_feats必须有80，所以要填充进去 然后记录下来
            ##有数据的特征 mask就是1
            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            # (Pdb) current_captions
            # array(['A person is slicing an onion.'], dtype=object)
            #print(current_captions, '\n')
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
            current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))
            
            # (Pdb) current_captions
            # ['<bos> A person is slicing an onion']
            # (Pdb) 
            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            # (Pdb) current_captions
            # ['<bos> A person is slicing an onion <eos>']    
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)
            # pdb.set_trace()

            decoder_input_data = np.zeros(
                (batch_size, n_caption_lstm_step+1, len(wordtoix)),
                dtype='float32')
            decoder_target_data = np.zeros(
                (batch_size, n_caption_lstm_step+1, len(wordtoix)),
                dtype='float32')
            for i,one_cap in enumerate(current_caption_ind):
                for j,one_word in enumerate(one_cap):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, j, one_word] = 1.
                    if j > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[i, j - 1, one_word] = 1.
            current_caption_matrix = tf.keras.preprocessing.sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x!=0).sum() + 1, current_caption_matrix)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            loss_val = model_train.train_on_batch([current_feats, decoder_input_data],decoder_target_data)
            loss_to_draw_epoch.append(loss_val[0])

            print('idx:', start, 'Epoch:', epoch, 'loss:', loss_val[0], "acc", loss_val[1], 'Elapsed time:', str((time.time()-start_time)))
            loss_fd.write('epoch ' + str(epoch) + 'loss ' + str(loss_val) + '\n')


        if np.mod(epoch, 10) == 0 and epoch > 0:
            print("Epoch ", epoch, "is done.")

        if np.mod(epoch,10) == 0 and epoch > 0:
            print("Epoch ", epoch, "is done. Saving the model...")
            model_train.save_weights("./model.h5")
    loss_fd.close()

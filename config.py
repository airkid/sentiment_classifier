#!/usr/bin/env python
# -*- coding: utf-8 -*-
class config():
    dim_word = 200
    dim_rnn = 256
    nlabels = 6
    learning_rate = 0.0001
    lr_decay = 0.9
    dropout = 0.5
    batch_size = 60
    nepoch = 20

    min_count = 2
    train_data = 'data/train.txt'
    split_ratio = 0.80
    test_data = 'data/test.txt'
    w2v_data = 'data/vector.txt'
    embedding_data = 'data/w2v.pkl'
    save_train = 'data/train.pkl'
    save_w2v = 'data/w2v.pkl'
    save_map = 'data/map.pkl'

    filewriter_path = './graph'
    output_path = './model'

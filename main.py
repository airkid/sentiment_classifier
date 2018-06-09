# -*- coding: utf-8 -*-
import os
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
from sentiment_model import sentimentModel
import data_utils
from config import config

def train():
    print 'loading data...'
    data = data_utils.load_data(config.save_train)
    train_data, dev_data = data_utils.data_split(data, config.split_ratio)
    print 'building_model'
    sen_model = sentimentModel(config)
    sen_model.build()
    print 'training'
    sen_model.train(train_data, dev_data)

def test():
    test_data = data_utils.load_data(config.test_data)
    model = sentimentModel(config)
    model.build()
    model.evaluate(test_data)


if __name__ == '__main__':
    data_utils.train_word2vec(config.train_data, config.w2v_data, config.dim_word, config.min_count)
    data_utils.generate_train(config.train_data, config.w2v_data, config.dim_word, config.save_train, config.save_w2v, config.save_map)
    train()
   # annotate()


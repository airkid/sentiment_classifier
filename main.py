# -*- coding: utf-8 -*-
import os
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
from sentiment_model import *
import data_utils

dim_word = 200
dim_lstm = 256
nlabels = 6
learning_rate = 0.0001
lr_decay = 0.9
dropout = 0.5
batch_size = 60
output_path = os.getcwd()
nepoch = 30
embedding = data_utils.load_data('w2v.pkl')
config = Config(embedding, dim_word, dim_lstm, nlabels, learning_rate, lr_decay, dropout, batch_size, output_path, nepoch)

def train():
    print 'loading data...'
    train_data = data_utils.load_data('train_data.pkl')
    dev_data = data_utils.load_data('dev_data.pkl')
    print 'building_model'
    model = sentimentModel(config)
    model.build()
    print 'training'
    model.train(train_data, dev_data)

def test():
    test_data = data_utils.load_data('test_data_2014.pkl')
    model = sentimentModel(config)
    model.build()
    model.evaluate(test_data)

def annotate():
    test_data = data_utils.load_data('test_data.pkl')
    src_test_data = data_utils.load_data('src_test_data.pkl')
    print 'building_model'
    model = sentimentModel(config)
    model.build()
    print 'annotate'
    num = len(test_data)
    pred_list = []
    score_list = []
    for i in xrange(num):
        src = src_test_data[i]
        ids = test_data[i]
        pred,score = model.annotate(ids)
        pred_list.append(pred)
	score_list.append(score)
    data_utils.save_data(pred_list,'crawl_pred.pkl')
    data_utils.save_data(score_list,'crawl_score.pkl')

if __name__ == '__main__':
    # train()
    test()
   # annotate()


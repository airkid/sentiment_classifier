# -*- coding: utf-8 -*-
'''
用于获取不同来源的数据，并将其转换为固定格式用于训练
'''
import random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
import copy
import json
import pickle
from gensim.models.word2vec import Word2Vec

def train_word2vec(datadir, savedir, dimword, min_count):
    print 'training word to vector model from ', datadir
    x = []
    datafile = codecs.open(datadir, 'r', 'utf-8')
    for line in datafile:
        x.append([word.strip() for word in line.split('\t')[1].strip().split(' ')])
    model = Word2Vec(x, size=dimword, min_count=min_count)
    model.save_word2vec_format(savedir ,binary=False)

def generate_dict(dictdir, dimword):
    dictfile = codecs.open(dictdir,'r','utf-8')
    num_map = {'UNK':0}
    w2v = [[0.0 for i in range(dimword)]]#unk
    total_num, dim = dictfile.readline().split(' ')
    for i in xrange(int(total_num)):
        if i % 10000 == 0:
            print i
        data = dictfile.readline()
        tmp = data.find(' ')
        word = data[:tmp]
        vec = data[tmp+1:-2].split(' ')
        vec = map(float, vec)
        num_map[word] = i+1
        w2v.append(vec)
    dictfile.close()
    return num_map, w2v

def transform_data(_data, num_map):
    data = copy.deepcopy(_data)
    idx_set = set(num_map.keys())
    missing_list = []
    for i in xrange(len(data)):
        #if i % 100000 == 0:
        #    print i
        for j in xrange(len(data[i])):
            if data[i][j] in idx_set:
                data[i][j] = num_map[data[i][j]]
            else:
                missing_list.append(data[i][j])
                data[i][j] = 0 #UNK
    return data, missing_list

def read_data(datadir):
    x,y = [[],[]]
    datafile = codecs.open(datadir,'r','utf-8')
    for line in datafile.readlines():
	label, sentence = line.split('\t')
	y.append(int(label))
        x.append([word.strip() for word in sentence.strip()])
    return x,y

def save_data(data,file_name):
    tmp = codecs.open(file_name,'wb')
    pickle.dump(data,tmp)
    tmp.close()

def load_data(file_name):
    tmp = codecs.open(file_name,'rb')
    data = pickle.load(tmp)
    tmp.close()
    return data

def combine(x,y):
    data = []
    for i in xrange(len(y)):
        data.append((x[i],y[i]))
    return data

def data_split(data, ratio):
    split_index = int(len(data) * ratio)
    train = data[:split_index]
    dev = data[split_index:]
    return train, dev

def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    for (x,y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def generate_train(traindir,w2vdir,dimword,save_train,save_w2v,save_map):
    print 'read'
    x,y = read_data(traindir)
    print 'generate'
    num_map, w2v = generate_dict(w2vdir, dimword)
    print 'transform'
    _x,missing_list = transform_data(x,num_map)
    data = combine(_x,y)
    print "total num: ", len(data)
    print 'saving'
    save_data(data,save_train)
    save_data(w2v,save_w2v)
    save_data(num_map,save_map)

def generate_test(testdir,w2vdir,dimword):
    x = read_data(testdir)
    #print len(x), x[0]
    num_map, w2v = generate_dict(w2vdir,dimword)
    trans_x = []
    for sentences in x:
        trans_x.append(transform_data(sentences,num_map)[0])
    save_data(x,'src_test_data.pkl')
    save_data(trans_x,'test_data.pkl')
    	
if __name__=='__main__':
    generate_train()

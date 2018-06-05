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

def read_data_ecm(datadir,numlabels):
    x,y = [[],[]]
    datafile = codecs.open(datadir,'r','utf-8')
    jsonfile = json.load(datafile)
    for idx,item in enumerate(jsonfile):
        for dataitem in item:
            #sentence
            sentence = dataitem[0].strip().split(' ')
            x.append([word.strip() for word in sentence])
            #label
            # y_tmp = [0 for i in range(numlabels)]
            # y_tmp[dataitem[1]] = 1
            y.append(int(dataitem[1]))
    datafile.close()
    return x,y

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

def read_data_NLPCC_2014(datadir):
    x,y = [[],[]]
    datafile = codecs.open(datadir,'r','utf-8')
    for line in datafile.readlines():
        label, sentence = line.split('\t')
	sentence = sentence.strip().split(' ')
        x.append([word.strip() for word in sentence])
        y.append(int(label))
    datafile.close()
    return x,y

def read_data_crawl(datadir):
    x = []
    datafile = codecs.open(datadir,'r','utf-8')
    for line in datafile.readlines():
	sentences = line.strip().split('\t')
        _x = []
	for sentence in sentences:
	    _x.append([word.strip() for word in sentence.strip()])
	x.append(_x[:])
    return x

def read_data_test(datadir):
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

def generate_train():
    print 'read'
    #x,y = read_data_ecm("data/train.json",6)
    #x,y = read_data_NLPCC_2014("data/2014_data.txt")
    x,y = read_data_test("data/test_data_2014.txt")
    #x.extend(_x)
    #y.extend(_y)
    #dev_x,dev_y = read_data_NLPCC_2014("data/dev.txt")
    print 'generate'
    num_map, w2v = generate_dict('data/vector_2014_full_200.txt',200)
    print len(w2v),w2v[1]
    print 'transform'
    x,missing_list = transform_data(x,num_map)
    #dev_x,dev_missing_list = transform_data(dev_x,num_map)
    #print 'missing_x: ',len(set(missing_list))
    #print 'missing_dev_x: ',len(set(dev_missing_list))
    test_data = combine(x,y)
    random.shuffle(test_data)
    #dev_data = combine(dev_x,dev_y)
    #random.shuffle(dev_data)
    #print len(train_data)
    #print len(dev_data)
    print 'saving'
    #save_data(train_data[:100000],'train_mini_data.pkl')
    save_data(test_data,'test_data_2014.pkl')
    #save_data(dev_data,'dev_data.pkl')
    #save_data(w2v,'w2v.pkl')
    #save_data(num_map,'map.pkl')

def generate_test():
    x = read_data_crawl('data/crawl.txt')  
    #print len(x), x[0]
    num_map, w2v = generate_dict('data/vector.txt',100)
    trans_x = []
    for sentences in x:
        trans_x.append(transform_data(sentences,num_map)[0])
    #print x[0]
    #print trans_x[0]
    save_data(x,'src_test_data.pkl')
    save_data(trans_x,'test_data.pkl')
    	
if __name__=='__main__':
    generate_train()

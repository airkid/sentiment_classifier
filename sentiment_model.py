# -*- coding: utf-8 -*-
'''
模型
'''
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
from data_utils import minibatches, load_data
import logging
from config import config

class sentimentModel(object):
    def __init__(self, config):
        self.config = config
        # self.logger = logger
        # if logger is None:
        #     logger = logging.getlogger('logger')
        #     logger.setLevel(logging.DEBUG)
        #     logging.basicConfig(format='%(message)s', lebel=logging.DEBUG)

    def add_placeholders(self):
        self.data_x = tf.placeholder(tf.int32, shape=[None, None])
        self.data_length = tf.placeholder(tf.int32, shape=[None])
        self.data_y = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder(tf.float32, shape=[])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(load_data(self.config.embedding_data), name="_word_embeddings", dtype=tf.float32, trainable=True)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.data_x, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.config.dropout)

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.dim_rnn)
            _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, self.word_embeddings, sequence_length=self.data_length, dtype=tf.float32)
            lstm_output = tf.concat((output_state_fw.h, output_state_bw.h), axis=-1)
            lstm_output = tf.nn.dropout(lstm_output, self.config.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2*self.config.dim_rnn, self.config.nlabels], dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.config.nlabels], dtype=tf.float32, initializer=tf.zeros_initializer())
        lstm_output = tf.reshape(lstm_output, [-1, 2*self.config.dim_rnn])
        pred = tf.matmul(lstm_output, W) + b
        self.logits=pred

    def add_pred_op(self): 
        self.score = tf.nn.softmax(self.logits)
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        _accuracy = tf.equal(self.labels_pred, self.data_y)
        self.acc = tf.reduce_mean(tf.cast(_accuracy,tf.float32))
        tf.summary.scalar("accuracy", self.acc)

    def add_loss_op(self):
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.data_y)
        self.loss = tf.reduce_mean(_loss)
        tf.summary.scalar("loss", self.loss)

    def add_train_op(self):
        with tf.variable_scope("train_op"):
            opt = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = opt.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.filewriter_path, sess.graph)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None, one=False):
        sequence_length = [len(sen) for sen in words]
        max_length = max(sequence_length)
        for i in xrange(len(words)):
            words[i].extend([0 for j in range(max_length-len(words[i]))])
        feed = {
            self.data_x: np.array(words),
            self.data_length: np.array(sequence_length)
        }
        if labels is not None:
            feed[self.data_y] = labels
        if lr is not None:
            feed[self.learning_rate] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        return feed, sequence_length

    def predict_batch(self, sess, words, labels):
        fd, sequence_length = self.get_feed_dict(words, labels=labels, dropout=1.0)
        pred, acc = sess.run([self.labels_pred, self.acc], feed_dict=fd)
        return pred, acc

    def run_eval(self, sess, test):
        accs = []
        correct_preds, total_correct, total_preds = 0,0,0
        for i, (words, labels) in enumerate(minibatches(test, self.config.batch_size)):
            pred,  _acc= self.predict_batch(sess, words, labels)
            accs.append(_acc)
        acc = np.mean(np.array(accs))
        return acc

    def run_epoch(self, sess, train, dev, epoch):
        nbatches = (len(train) + self.config.batch_size -1) / self.config.batch_size
        total_loss = 0.0
        batch_cnt = 0
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels=labels, lr=self.config.learning_rate, dropout=self.config.dropout)
            _, loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            total_loss += loss
            if i % 100000 == 0:
                acc = self.run_eval(sess,dev)
                self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_acc',simple_value=acc)]),epoch)
            batch_cnt += 1
        if i % 20 == 0:
            self.file_writer.add_summary(summary, epoch * nbatches + i)
            if i % 1000 == 0:
                print("batch {}, loss {:04.2f}.".format(i, float(total_loss)/batch_cnt))
        acc = self.run_eval(sess, dev)
    #self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_acc',simple_value=acc)]),epoch)
        print("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * 0))
        return acc

    def train(self, train, dev):
        best_score = 0
        saver = tf.train.Saver()
        early_stopping_round = -1
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.config.output_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt)
                saver.resote(sess, self.config.output_path)
            else:
                print('Begin to initialize ...')
                sess.run(self.init)
            self.add_summary(sess)
            for epoch in range(self.config.nepoch):
                print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepoch))
                
                acc = self.run_epoch(sess, train, dev, epoch)
                
                if acc > best_score:
                    if not os.path.exists(self.config.output_path):
                        os.makedirs(self.config.output_path)
                    saver.save(sess, self.config.output_path)
                    best_score = acc
                    print("- new best score! ",acc)

                self.config.learning_rate *= self.config.lr_decay
    
    def evaluate(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Testing model over test set")
            saver.restore(sess, self.config.output_path)
            acc = self.run_eval(sess, test)
            print("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * 0))

 #    def annotate(self, test):
    # #print test
    # saver = tf.train.Saver()
 #        pred_list = []
    # score_list = []
    # with tf.Session() as sess:
    #     saver.restore(sess, self.config.output_path)
    #     for words in test:
    #     fd, sequence_length = self.get_feed_dict([words],  dropout=1.0)
 #                pred, score = sess.run([self.labels_pred, self.score], feed_dict=fd)
    #     pred_list.append(pred[0])
    #     score_list.append(score[0][pred[0]])
    # return pred_list,score_list 

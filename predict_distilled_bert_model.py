from bert4sqq.models import build_model
from bert4sqq.backend import keras, K
from bert4sqq.tokenizers import FullTokenizer
from bert4sqq.snippets import sequence_padding, DataGenerator
from tensorflow.python.keras.utils import losses_utils
from keras.layers import Lambda
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import os
import random
from keras.callbacks import ModelCheckpoint
import time

###### 下面是蒸馏训练

config_path = '../models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../models/chinese_L-12_H-768_A-12/vocab.txt'
input_data_path = './input'

maxlen = 512
batch_size = 8
epochs = 1
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
speed = 0.5

checkpoint_path = './output/distill-0001.ckpt'

main_labels = ['com.sqq.hy.music', 'com.sqq.hy.iptv', 'com.sqq.audiocontent']
id2label = dict(enumerate(main_labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(main_labels)

tokenizer = FullTokenizer(vocab_file=dict_path)

fast_bert_class_model = build_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                    model='fast_bert', labels_num=num_labels,
                                    distillate_or_not=True)


for (i, layer) in enumerate(fast_bert_class_model.layers):
    print(layer.name)

fast_bert_class_model.summary()


def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    inputs = model.input + [K.learning_phase()]
    output = model.layers[index].output
    layerss = K.function(inputs, [output], name='predict_function')
    ret = layerss(x + [1])
    ret_output = ret[0]
    ret_output = np.squeeze(ret_output)
    return ret_output


def cus_predict(eval_model, input_x):
    # 200是Classifier-Transformer-Classifier-MultiHeadSelfAttention-Layer0-Norm
    layer_start_index = 200
    for i in range(12):
        start_time = time.clock()
        layer_out = get_layer_output(eval_model, input_x, index=layer_start_index + i)
        stop_time = time.clock()
        cost = stop_time - start_time
        print(cost)
    return layer_out


tokens = tokenizer.tokenize('我想听刘德华的歌')

input_ids = tokenizer.convert_tokens_to_ids(tokens)
segment_ids = [0] * len(input_ids)
input_ids = np.array(input_ids)
segment_ids = np.array(segment_ids)
input_ids = np.expand_dims(input_ids, 0)
segment_ids = np.expand_dims(segment_ids, 0)

output = cus_predict(fast_bert_class_model, [input_ids, segment_ids])


tokens = tokenizer.tokenize('听一下周杰伦的青花瓷')

input_ids = tokenizer.convert_tokens_to_ids(tokens)
segment_ids = [0] * len(input_ids)
input_ids = np.array(input_ids)
segment_ids = np.array(segment_ids)
input_ids = np.expand_dims(input_ids, 0)
segment_ids = np.expand_dims(segment_ids, 0)

output = cus_predict(fast_bert_class_model, [input_ids, segment_ids])
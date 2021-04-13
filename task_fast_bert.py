# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert4sqq.models import build_model
from bert4sqq.backend import keras, K
from bert4sqq.tokenizers import FullTokenizer
from bert4sqq.snippets import sequence_padding, DataGenerator
from bert4sqq.utils import get_labels, load_data
from bert4sqq.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from scipy import stats
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
import random
import tensorflow as tf
from keras.utils.np_utils import to_categorical

config_path = '../models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../models/chinese_L-12_H-768_A-12/vocab.txt'
input_data_path = './input'
checkpoint_save_path = "./output/cp-{epoch:04d}.ckpt"
output_dir = './output'
file_name = 'train-domain.txt'

maxlen = 512
batch_size = 64
epochs = 10
learning_rate = 5e-5  # bert_layers越小，学习率应该要越大

main_labels = get_labels(input_data_path, output_dir, file_name)
id2label = dict(enumerate(main_labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(main_labels)


def convert_id_to_label(pred_ids_result):
    max_probabilities = -1
    max_id = 0
    for i, probabilities in enumerate(pred_ids_result):
        if max_probabilities < probabilities:
            max_probabilities = probabilities
            max_id = i
    curr_label = id2label[max_id]
    return curr_label


train_data = load_data(input_data_path, file_name)
total_length = len(train_data)
test_length = int(total_length * 5 / 100)

valid_data = train_data[:test_length]
# train_data = train_data[test_length:]
test_data = valid_data
# random.shuffle(train_data)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            splits = item.strip().split("\t")
            input_str = splits[0].strip()
            input_label = splits[-1].strip()
            tokens = tokenizer.tokenize(input_str)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            # labels = to_categorical(label2id.get(input_label), num_labels)
            labels = [label2id.get(input_label)]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


tokenizer = FullTokenizer(vocab_file=dict_path)

fast_bert = build_model(config_path=config_path, checkpoint_path=checkpoint_path, model='fast_bert',
                        labels_num=num_labels, return_keras_model=False)

# bert = build_model(config_path=config_path, checkpoint_path=checkpoint_path, model='bert', labels_num=num_labels, return_keras_model=False)

# output = Lambda(lambda x: x[:, 0])(fast_bert.model.output)
# output = Dense(
#     units=num_labels,
#     activation='softmax',
#     kernel_initializer=fast_bert.initializer
# )(output)
#
# model = Model(fast_bert.model.input, output)

output = Dense(
    units=num_labels,
    activation='softmax',
    kernel_initializer=fast_bert.initializer
)(fast_bert.output)

model = Model(fast_bert.model.input, output)

# model = Model(fast_bert.model.input, fast_bert.output)
model.summary()

model.compile(
    # 输出结果是求平均后的结果
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    # loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['sparse_categorical_accuracy'],
)

# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

def recognize(text, eval_model):
    tokens = tokenizer.tokenize(text)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    output = eval_model.predict([[input_ids], [segment_ids]])[0].argmax()
    label_ret = id2label[output]
    return label_ret


def evaluate(data, eval_model):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for item in tqdm(data):
        splits = item.strip().split("\t")
        input_str = splits[0].strip()
        input_label = splits[-1].strip()
        result = recognize(input_str, eval_model)

        if result == input_label:
            X += 1
        Y += 1
        Z += 1
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, eval_model, save_model=False):
        self.best_val_f1 = 0
        self.eval_model = eval_model
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data, self.eval_model)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            if self.save_model:
                self.eval_model.save_weights('./output/best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data, self.eval_model)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


evaluator = Evaluator(model)
train_generator = data_generator(train_data, batch_size)

# 创建callback来保存模型的权重
cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                              save_weights_only=True,
                              verbose=1)

model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    # callbacks=[evaluator]
    callbacks=[cp_callback, evaluator]
)

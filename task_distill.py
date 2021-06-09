# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert4sqq.models import build_model
from bert4sqq.backend import keras, K
from bert4sqq.tokenizers import FullTokenizer
from bert4sqq.snippets import sequence_padding, DataGenerator
from tensorflow.python.keras.utils import losses_utils
from keras.layers import Lambda
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from bert4sqq.utils import get_labels2id, load_data
import random
from keras.callbacks import ModelCheckpoint

###### 下面是蒸馏训练

config_path = '../models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../models/chinese_L-12_H-768_A-12/vocab.txt'
input_data_path = './input'
checkpoint_save_path = "./output/distill-{epoch:04d}.ckpt"
labels_dir = './output'
file_name = 'train-domain.txt'

batch_size = 8
epochs = 1
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
speed = 0.5

checkpoint_path = './output/cp-0001.ckpt'

label2id = get_labels2id(labels_dir)
id2label = {value: key for key, value in label2id.items()}
num_labels = len(id2label)

tokenizer = FullTokenizer(vocab_file=dict_path)

fast_bert_class_model = build_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                    model='fast_bert', labels_num=num_labels,
                                    distillate_or_not=True)


def lo(x):
    kl = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE)
    teacher_probs = K.softmax(x[-1])
    loss = 0
    for i in range(len(x) - 1):
        student_logits = x[i]
        # student_probs = tf.nn.log_softmax(student_logits, axis=-1)
        student_probs = K.softmax(student_logits, axis=-1)
        # 需要实现连续分布的KL散度
        kl_value = kl(student_probs, teacher_probs)
        # ss = tf.less(kl_value, tf.constant(6, dtype=tf.float32))
        # if ss is True:
        #     print("111")
        loss += kl_value
    return loss


def convert_id_to_label(pred_ids_result):
    max_probabilities = -1
    max_id = 0
    for i, probabilities in enumerate(pred_ids_result):
        if max_probabilities < probabilities:
            max_probabilities = probabilities
            max_id = i
    curr_label = id2label[max_id]
    return curr_label


output = Lambda(lambda x: lo(x))(fast_bert_class_model.output)

fast_bert_class_model = Model(fast_bert_class_model.input, output)

for (i, layer) in enumerate(fast_bert_class_model.layers):
    # print(layer.name)
    if i < 104:
        layer.trainable = False
    elif 'Classifier' in layer.name and 'Layer11' in layer.name:
        layer.trainable = False
fast_bert_class_model.summary()

fast_bert_class_model.compile(
    loss=lambda y_true, y_pred: y_pred,
    # loss_weights=[1]*12,
    # loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    # metrics=['sparse_categorical_accuracy'],
    # metrics=[lambda y_true,y_pred: y_pred],
)

train_data = load_data(input_data_path, file_name)
random.shuffle(train_data)


# def get_layer_output(model, x, index=-1):
#     """
#     get the computing result output of any layer you want, default the last layer.
#     :param model: primary model
#     :param x: input of primary model( x of model.predict([x])[0])
#     :param index: index of target layer, i.e., layer[23]
#     :return: result
#     """
#     inputs = model.input + [K.learning_phase()]
#     output = model.layers[index].output
#     layerss = K.function(inputs, [output], name='predict_function')
#     ret = layerss(x + [1])
#     ret_output = ret[0]
#     ret_output = np.squeeze(ret_output)
#     return ret_output


# def cus_predict(eval_model, input_x):
#     # 200是Classifier-Transformer-Classifier-MultiHeadSelfAttention-Layer11-Norm
#     layer_start_index = 200
#     for i in range(12):
#         layer_out = get_layer_output(eval_model, input_x, index=layer_start_index + i)
#         # 熵越大则不确定性越大,则结果越可能是错的
#         output_probs = entropy(layer_out)
#         if output_probs < speed:
#             # 如果熵比较小，则认为预测结果可信
#             break
#     return layer_out
#
#
# def c_recognize(text, eval_model):
#     tokens = tokenizer.tokenize(text)
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     segment_ids = [0] * len(input_ids)
#     input_ids = np.array(input_ids)
#     segment_ids = np.array(segment_ids)
#     input_ids = np.expand_dims(input_ids, 0)
#     segment_ids = np.expand_dims(segment_ids, 0)
#
#     # x = model.predict([input_ids, segment_ids])
#     # output = eval_model.predict([input_ids, segment_ids])[0]
#     output = cus_predict(eval_model, [input_ids, segment_ids])
#     label_ret = convert_id_to_label(output)
#     return label_ret


# def c_evaluate(data, eval_model):
#     """评测函数
#     """
#     X, Y, Z = 1e-10, 1e-10, 1e-10
#     for item in tqdm(data):
#         splits = item.strip().split("\t")
#         input_str = splits[0].strip()
#         input_label = splits[-1].strip()
#         result = c_recognize(input_str, eval_model)
#
#         if result == input_label:
#             X += 1
#         Y += 1
#         Z += 1
#     f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
#     return f1, precision, recall


# class C_Evaluator(keras.callbacks.Callback):
#     """评估与保存
#     """
#
#     def __init__(self, eval_model, save_model=False):
#         self.best_val_f1 = 0
#         self.eval_model = eval_model
#         self.save_model = save_model
#
#     def on_epoch_end(self, epoch, logs=None):
#         f1, precision, recall = c_evaluate(valid_data, self.eval_model)
#         # 保存最优
#         if f1 >= self.best_val_f1:
#             self.best_val_f1 = f1
#             if self.save_model:
#                 self.eval_model.save_weights('./output/best_model.weights')
#         print(
#             'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
#             (f1, precision, recall, self.best_val_f1)
#         )
#         f1, precision, recall = c_evaluate(test_data, self.eval_model)
#         print(
#             'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
#             (f1, precision, recall)
#         )


# c_evaluator = C_Evaluator(fast_bert_class_model, True)


class c_data_generator(DataGenerator):
    """蒸馏数据生成器
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
            labels = [label2id.get(input_label)]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                # yield [batch_token_ids, batch_segment_ids], [batch_labels] * 12
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = c_data_generator(train_data, batch_size)

cp_callback = ModelCheckpoint(filepath=checkpoint_save_path,
                              save_weights_only=True,
                              verbose=1)

fast_bert_class_model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    # callbacks=[c_evaluator]
    callbacks=[cp_callback]
)

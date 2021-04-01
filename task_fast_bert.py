from bert4sqq.models import build_model
from bert4sqq.backend import keras, K
from bert4sqq.tokenizers import FullTokenizer
from bert4sqq.snippets import sequence_padding, DataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense
import tensorflow as tf
from scipy import stats
import os
from tqdm import tqdm

config_path = '../models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../models/chinese_L-12_H-768_A-12/vocab.txt'

maxlen = 512
batch_size = 8
epochs = 10
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大

main_labels = ['com.sqq.hy.music', 'com.sqq.hy.iptv', 'com.sqq.audiocontent']
id2label = dict(enumerate(main_labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(main_labels)

def load_data(data_dir):
    lines = []
    with open(os.path.join(data_dir, 'train-domain.txt'), encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            lines.append(line)
    return lines


train_data = load_data('./input')
total_length = len(train_data)
test_length = int(total_length * 5 / 100)

valid_data = train_data[:test_length]
train_data = train_data[test_length:]
test_data = valid_data


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

# model = build_model(config_path=config_path, checkpoint_path=checkpoint_path, model='fast_bert', labels_num=num_labels)
model = build_model(config_path=config_path, checkpoint_path=checkpoint_path, model='bert', labels_num=num_labels)

output = Dense(num_labels)(model.output)
model = Model(model.input, output)

model.summary()


def kl_for_log_probs(log_p, log_q):
    log_p_n = K.eval(log_p)
    log_q_n = K.eval(log_q)
    KL = stats.entropy(log_p_n, log_q_n, axis=-1)
    kl = K.mean(K.constant(KL), axis=0)
    return kl


def cross_entropy(y_true, y_pred):
    """计算fast_bert的loss
    """
    if not isinstance(y_true, list):
        # 如果是正常训练
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    else:
        # 如果是蒸馏
        teacher_probs = K.softmax(y_true[0])
        loss = 0
        for i in range(len(y_true) - 1):
            student_logits = y_true[i+1]
            # student_probs = K.log(K.softmax(student_logits))
            student_probs = tf.nn.log_softmax(student_logits, axis=-1)
            # 需要实现连续分布的KL散度
            loss += kl_for_log_probs(student_probs, teacher_probs)
    return cross_entropy


model.compile(
    # loss=cross_entropy,
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['sparse_categorical_accuracy'],
)


def recognize(text):
    tokens = tokenizer.tokenize(text)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    output = model.predict([input_ids, segment_ids])[0]



def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for item in tqdm(data):
        splits = item.strip().split("\t")
        input_str = splits[0].strip()
        input_label = splits[-1].strip()
        result = recognize(input_str)

        if result == input_label:
            X += 1
        Y += 1
        Z += 1
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


evaluator = Evaluator()
train_generator = data_generator(train_data, batch_size)

model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)
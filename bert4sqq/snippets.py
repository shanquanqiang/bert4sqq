# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import keras
import six
import numpy as np
import re

# six.PY2 返回一个表示当前运行环境是否为python2的boolean值
# six.PY3 返回一个表示当前运行环境是否为python3的boolean值
is_py2 = six.PY2
is_py3 = six.PY3
_open_ = open

if not is_py2:
    basestring = str


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


def strQ2B(ustring):
    """全角符号转对应的半角符号
    """
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def string_matching(s, keywords):
    """判断s是否至少包含keywords中的至少一个字符串
    """
    for k in keywords:
        if re.search(k, s):
            return True
    return False


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input.
        字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode(encoding, errors)
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif is_py2:
        if isinstance(text, str):
            return text.decode(encoding, errors)
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.encode(encoding, errors=errors)
    return text


class open:
    """模仿python自带的open函数，主要是为了同时兼容py2和py3
        一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
        虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。
        yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。
    """

    def __init__(self, name, mode='r', encoding=None, errors='ignore'):
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors
        # self.iterator = None

    def __iter__(self):
        return self
        # for l in self.file:
        #     if self.encoding:
        #         l = convert_to_unicode(l, self.encoding, self.errors)
        #     yield l

    def next(self):
        for in_line in self.file:
            if self.encoding:
                ret_line = convert_to_unicode(in_line, self.encoding, self.errors)
                return ret_line

    def __next__(self):
        return self.next()

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

def is_one_of(x, ys):
    """判断x是否在ys之中
    等价于x in ys，但有些情况下x in ys会报错
    """
    for y in ys:
        if x is y:
            return True
    return False

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d

def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


if __name__ == "__main__":
    with open('../t.txt', 'w', encoding='utf-8') as writer:
        writer.write('dsdddddd')
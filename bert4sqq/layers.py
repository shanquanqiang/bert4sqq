# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import *
from keras import initializers, activations
from bert4sqq.backend import keras, K, is_tf_keras
from bert4sqq.backend import sequence_masking
from bert4sqq.backend import recompute_grad


def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """

    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func


if (not is_tf_keras) and keras.__version__ < '2.3':

    class Layer(keras.layers.Layer):
        """重新定义Layer，赋予“层中层”功能
        （仅keras 2.3以下版本需要）
        """

        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

        def __setattr__(self, name, value):
            if isinstance(value, keras.layers.Layer):
                if not hasattr(self, '_layers'):
                    self._layers = []
                if value not in self._layers:
                    self._layers.append(value)
            super(Layer, self).__setattr__(name, value)

        @property
        def trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            if trainable:
                trainable_weights = super(Layer, self).trainable_weights[:]
                for l in getattr(self, '_layers', []):
                    trainable_weights += l.trainable_weights
                return trainable_weights
            else:
                return []

        @property
        def non_trainable_weights(self):
            trainable = getattr(self, 'trainable', True)
            non_trainable_weights = super(Layer, self).non_trainable_weights[:]
            for l in getattr(self, '_layers', []):
                if trainable:
                    non_trainable_weights += l.non_trainable_weights
                else:
                    non_trainable_weights += l.weights
            return non_trainable_weights


    if keras.__version__ < '2.2.5':

        import inspect


        class Model(keras.models.Model):
            """重新定义Model，整合fit和fit_generator
            """

            def fit(self, x=None, *args, **kwargs):
                if inspect.isgenerator(x):
                    return self.fit_generator(x, *args, **kwargs)
                else:
                    return super(Model, self).fit(x, *args, **kwargs)


        keras.models.Model = Model

else:

    class Layer(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(Layer, self).__init__(**kwargs)
            self.supports_masking = True  # 本项目的自定义层均可mask

if (not is_tf_keras) or tf.__version__ < '1.15':

    if not is_tf_keras:
        NodeBase = keras.engine.base_layer.Node
    else:
        from tensorflow.python.keras.engine import base_layer

        NodeBase = base_layer.Node


    class Node(NodeBase):
        """修改Node来修复keras下孪生网络的bug
        注意：这是keras的bug，并不是bert4keras的bug，但keras已经不更新了，
              所以只好在这里进行修改。tf 1.15+自带的keras已经修改了这个
              bug。
        """

        @property
        def arguments(self):
            return self._arguments.copy()

        @arguments.setter
        def arguments(self, value):
            self._arguments = value or {}


    if not is_tf_keras:
        keras.engine.base_layer.Node = Node
    else:
        base_layer.Node = Node


class Embedding(keras.layers.Embedding):
    """拓展Embedding层
    """

    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if K.ndim(inputs) == 2:
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """新增mode参数，可以为embedding或dense。如果为embedding，
        则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        """关于判据，本来是通过缓存call时的mode参数来判断的，但是后来发现
        Keras在使用compute_output_shape的时候不一定配套调用了call函数，
        所以缓存的mode可能是不准的，因此只能出此下策。
        """
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (K.int_shape(self.embeddings)[0],)


class BiasAdd(Layer):
    """加上偏置项
    """

    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)


class Concatenate1D(Layer):
    """1维序列拼接层
    说明：本来该功能可以直接通过Concatenate层来实现，无奈Keras
          自带的Concatenate层的compute_mask写得不合理，导致一个
          带mask的序列与一个不带mask的序列拼接会报错，因此干脆
          自己重写一个好了。
    """

    def call(self, inputs):
        return K.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    m = K.ones_like(inputs[i][..., 0], dtype='bool')
                masks.append(m)
            return K.concatenate(masks, axis=1)

    def compute_output_shape(self, input_shape):
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
            return (input_shape[0][0], seq_len, input_shape[0][2])
        else:
            return (input_shape[0][0], None, input_shape[0][2])


class MultiHeadAttention(Layer):
    """多头注意力机制
    """

    def __init__(
            self,
            heads,
            head_size,
            out_dim=None,
            key_size=None,
            use_bias=True,
            attention_scale=True,
            return_attention_scores=False,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            if mask[0] is not None:
                q_mask = K.cast(mask[0], K.floatx())
            if mask[2] is not None:
                v_mask = K.cast(mask[2], K.floatx())
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)
        # 返回结果
        o = sequence_masking(o, q_mask, 0)
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的atttention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size ** 0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, 1, -1)
        A = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            conditional=False,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer='glorot_uniform',
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            merge_mode='add',
            hierarchical=None,
            embeddings_initializer='zeros',
            custom_position_ids=False,
            **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul']:
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """

    def __init__(
            self,
            units,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'Embedding': Embedding,
    'BiasAdd': BiasAdd,
    'Concatenate1D': Concatenate1D,
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'PositionEmbedding': PositionEmbedding,
    # 'SinusoidalPositionEmbedding': SinusoidalPositionEmbedding,
    # 'RelativePositionEmbedding': RelativePositionEmbedding,
    # 'RelativePositionEmbeddingT5': RelativePositionEmbeddingT5,
    'FeedForward': FeedForward,
    # 'ConditionalRandomField': ConditionalRandomField,
    # 'MaximumEntropyMarkovModel': MaximumEntropyMarkovModel,
    # 'Loss': Loss,
}

keras.utils.get_custom_objects().update(custom_objects)

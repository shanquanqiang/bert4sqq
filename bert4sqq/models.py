# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from bert4sqq.layers import *
from bert4sqq.snippets import is_string
from keras.models import Model
from scipy import stats


class Transformer(object):
    """模型基类
    """

    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate=None,  # Dropout比例
            embedding_size=None,  # 是否指定embedding_size
            attention_head_size=None,  # Attention中V的head_size
            attention_key_size=None,  # Attention中Q,K的head_size
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            layers=None,  # 外部传入的Keras层
            prefix=None,  # 层名前缀
            name=None,  # 模型名称
            **kwargs
    ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    def build(
            self,
            attention_caches=None,
            layer_norm_cond=None,
            layer_norm_cond_hidden_size=None,
            layer_norm_cond_hidden_act=None,
            additional_input_layers=None,
            **kwargs
    ):
        """模型构建函数
        attention_caches：为Attention的K,V的缓存序列字典，格式为
                         {Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数：实现Conditional Layer Normalization时使用，
                            用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        # Call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """通过apply调用层会自动重用同名层
        inputs: 上一层的输出；
        layer: 要调用的层类名；
        arguments: 传递给layer.call的参数；
        kwargs: 传递给层初始化的参数。
        """
        layer_trainable = kwargs.get('layer_trainable')
        if layer_trainable is None:
            layer_trainable = True
        else:
            kwargs.pop('layer_trainable')

        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        if layer is MultiHeadAttention and self.residual_attention_scores:
            kwargs['return_attention_scores'] = True

        arguments = arguments or {}
        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name

            # 设置该层是不是需要训练，不是蒸馏的时候主干设置为可训练，蒸馏的时候主干需要冻结
            layer.trainable = layer_trainable
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name], MultiHeadAttention):
                if name in self.attention_caches:
                    # 如果检测到Cache的传入，那么自动在Key,Value处拼接起来
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', name + '-Cached-Value'
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=v_name)([v_cache, inputs[2]])
                    inputs = inputs[:1] + [k, v] + inputs[3:]
                if self.residual_attention_scores:
                    # 如果使用残差Attention矩阵，则给每个Attention矩阵加上前上一层的Attention
                    # 矩阵，这对应RealFormer设计（https://arxiv.org/abs/2012.11747）。目前
                    # 该实现还相对粗糙，可能欠缺通用性。
                    if self.attention_scores is not None:
                        if arguments.get('a_bias'):
                            a_bias = Add(name=name + '-Attention-Bias'
                                         )([inputs[3], self.attention_scores])
                        else:
                            a_bias = self.attention_scores
                        inputs = inputs[:3] + [a_bias] + inputs[4:]
                        arguments['a_bias'] = True
                    o, a = self.layers[name](inputs, **arguments)
                    self.attention_scores = a
                    return o
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    item = (item, [1] * len(item))
                ext_embeddings.append(
                    np.average(embeddings[item[0]], 0, item[1])
                )
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or K.floatx()
        return K.variable(
            self.initializer(value.shape, dtype), dtype, name=name
        ), value

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载bert权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            layer = self.layers[layer]
            weights = layer.trainable_weights
            values = [self.load_variable(checkpoint, v) for v in variables]

            if isinstance(layer, MultiHeadAttention):
                """如果key_size不等于head_size，则可以通过
                正交矩阵将相应的权重投影到合适的shape。
                """
                count = 2
                if layer.use_bias:
                    count += 2
                heads = self.num_attention_heads
                head_size = self.attention_head_size
                key_size = self.attention_key_size
                W = np.linalg.qr(np.random.randn(key_size, head_size))[0].T
                if layer.attention_scale:
                    W = W * key_size ** 0.25 / head_size ** 0.25
                for i in range(count):
                    w, v = weights[i], values[i]
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if w_shape[-1] != v_shape[-1]:
                        pre_shape = w_shape[:-1]
                        v = v.reshape(pre_shape + (heads, head_size))
                        v = np.dot(v, W)
                        v = v.reshape(pre_shape + (heads * key_size,))
                        values[i] = v

            weight_value_pairs.extend(zip(weights, values))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None, dtype=None):
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    variable, value = self.create_variable(name, value, dtype)
                    all_variables.append(variable)
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)


class BERT(Transformer):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            hierarchical_position=None,  # 是否层次分解位置编码
            custom_position_ids=False,  # 是否自行传入位置id
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position'
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(BERT, self).load_variable(checkpoint, name)
        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            return self.load_embeddings(variable)
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(BERT, self).create_variable(name, value, dtype)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


class FAST_BERT(Transformer):
    """构建FAST BERT模型
    """

    def __init__(
            self,
            labels_num,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            hierarchical_position=None,  # 是否层次分解位置编码
            custom_position_ids=False,  # 是否自行传入位置id
            shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
            cla_attention_head_size=None,
            cla_attention_key_size=None,
            pooling='first',
            distillate_or_not=False,
            predict=False,
            speed=0.5,  # 不确定度阈值，如果如果预测结果熵小于此阈值则表示此结果可信。熵越大则不确定性越大,则结果越可能是错的
            **kwargs  # 其余参数
    ):
        super(FAST_BERT, self).__init__(**kwargs)
        self.labels_num = labels_num
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.cla_attention_head_size = cla_attention_head_size or self.cla_hidden_size // self.cla_heads_num
        self.cla_attention_key_size = cla_attention_key_size or self.cla_attention_head_size
        self.pooling = pooling
        # self.predict = predict
        self.distillate_or_not = distillate_or_not
        # self.distillate_or_not = self.predict if self.predict else distillate_or_not
        # self.layer_trainable = False if self.distillate_or_not is True else True
        self.layer_trainable = True
        # self.hidden_list = []
        self.speed = speed

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        distill = []
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
            if self.distillate_or_not:
                # 如果是蒸馏就在每一层transformer后面加classifier层
                x = self.apply_distill_classifier_layer(outputs, i, True)
                # if self.predict:
                #     # 如果是预测
                #     probs = self.entropy(x)
                #     ss = tf.less(probs, tf.constant(self.speed, dtype=tf.float32))
                #     if ss is True:
                #         # 如果熵比较小，则认为预测结果可信
                #         outputs = x
                #         break
                # else:
                #     # 如果不是预测，则将此结果也输出
                distill.append(x)
            else:
                # 如果不是蒸馏正常添加最后一层classifier
                if i == self.num_hidden_layers - 1:
                    outputs = self.apply_distill_classifier_layer(outputs, i, True)
        if distill:
            outputs = distill
        #
        # if len(outputs) == 1:
        #     outputs = outputs[0]

        return outputs

    def get_inputs(self):
        """FAST BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token',
            layer_trainable=self.layer_trainable
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment',
                layer_trainable=self.layer_trainable
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position',
                layer_trainable=self.layer_trainable
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self, inputs):
        """FAST BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token',
            layer_trainable=self.layer_trainable
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name,
                layer_trainable=self.layer_trainable
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment',
                layer_trainable=self.layer_trainable
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position',
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm',
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout',
            layer_trainable=self.layer_trainable
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping',
                layer_trainable=self.layer_trainable
            )

        return x

    def apply_main_layers(self, inputs, index):
        """FAST BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name,
            layer_trainable=self.layer_trainable
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name,
            layer_trainable=self.layer_trainable
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name,
            layer_trainable=self.layer_trainable
        )

        return x

    def apply_distill_classifier_layer(self, inputs, index, layer_trainable):
        """ hidden_size = 128 heads_num = 2
        """
        z = self.layer_norm_conds[0]
        x = inputs
        outputs = [x]

        x = outputs[0]

        x = self.apply(
            inputs=x,
            layer=Dense,
            units=self.cla_hidden_size,
            activation='tanh',
            kernel_initializer=self.initializer,
            name='Classifier-Dense-One-Layer%d' % index,
            layer_trainable=layer_trainable
        )

        xi, x, arguments = x, [x, x, x], {'a_bias': None}

        cla_attention_name = 'Transformer-Classifier-MultiHeadSelfAttention-Layer%d' % index
        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.cla_heads_num,
            head_size=self.cla_attention_head_size,
            out_dim=self.cla_hidden_size,
            key_size=self.cla_attention_key_size,
            kernel_initializer=self.initializer,
            name=cla_attention_name,
            layer_trainable=layer_trainable
        )

        # x = self.apply(
        #     inputs=x,
        #     layer=Dropout,
        #     rate=self.dropout_rate,
        #     name='%s-Dropout' % cla_attention_name,
        #     layer_trainable=layer_trainable
        # )
        #
        # x = self.apply(
        #     inputs=[xi, x], layer=Add, name='%s-Add' % cla_attention_name,
        #     layer_trainable=layer_trainable
        # )
        #
        # x = self.apply(
        #     inputs=self.simplify([x, z]),
        #     layer=LayerNormalization,
        #     conditional=(z is not None),
        #     hidden_units=self.layer_norm_conds[1],
        #     hidden_activation=self.layer_norm_conds[2],
        #     hidden_initializer=self.initializer,
        #     name='%s-Norm' % cla_attention_name,
        #     layer_trainable=layer_trainable
        # )

        if self.pooling == "mean":
            # 这里可能有问题
            x = K.mean(x, dim=-1)
        elif self.pooling == "max":
            # 这里可能有问题
            x = K.max(x, dim=1)[0]
        elif self.pooling == "last":
            x = Lambda(lambda y: y[:, -1, :])(x)
        else:
            x = Lambda(lambda y: y[:, 0, :])(x)

        x = self.apply(
            inputs=x,
            layer=Dense,
            units=self.cla_hidden_size,
            activation='tanh',
            kernel_initializer=self.initializer,
            name='Classifier-Dense-Two-Layer%d' % index,
            layer_trainable=layer_trainable
        )

        x = self.apply(
            inputs=x,
            layer=Dense,
            units=self.labels_num,
            kernel_initializer=self.initializer,
            name='Classifier-Dense-Three-Layer%d' % index,
            layer_trainable=layer_trainable
        )

        # x = self.apply(
        #     inputs=self.simplify([x, z]),
        #     layer=LayerNormalization,
        #     conditional=(z is not None),
        #     hidden_units=self.layer_norm_conds[1],
        #     hidden_activation=self.layer_norm_conds[2],
        #     hidden_initializer=self.initializer,
        #     name='Classifier-%s-Norm' % cla_attention_name,
        #     layer_trainable=layer_trainable
        # )

        outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

    def entropy(self, output):
        # 熵越大则不确定性越大
        # 计算概率分布
        probs = tf.nn.softmax(output)
        # probs_n = K.eval(probs)
        # # 计算底数为base的熵
        # en = stats.entropy(probs_n, base=self.labels_num)
        # return en
        return tf.constant(6, dtype=tf.float32)

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        variable = super(FAST_BERT, self).load_variable(checkpoint, name)

        if name in [
            'bert/embeddings/word_embeddings',
            'cls/predictions/output_bias',
        ]:
            return self.load_embeddings(variable)
        elif name == 'cls/seq_relationship/output_weights':
            return variable.T
        else:
            return variable

    def create_variable(self, name, value, dtype=None):
        """在tensorflow中创建一个变量
        """
        if name == 'cls/seq_relationship/output_weights':
            value = value.T
        return super(FAST_BERT, self).create_variable(name, value, dtype)

    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping


def build_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        return_keras_model=True,
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)

    models = {
        'bert': BERT,
        'fast_bert': FAST_BERT,
    }

    if is_string(model):
        model = model.lower()
        chosen_model = models[model]
    else:
        chosen_model = model

    # 这个参数暂时用不到
    application = application.lower()

    model_in_using = chosen_model(**configs)
    model_in_using.build(**configs)

    distillate_or_not = configs.get('distillate_or_not', False)
    # predict = configs.get('predict', False)
    # distillate_or_not = predict if predict else distillate_or_not
    if checkpoint_path is not None:
        if distillate_or_not is False:
            model_in_using.load_weights_from_checkpoint(checkpoint_path)
        else:
            model_in_using.model.load_weights(checkpoint_path, by_name=True)

    if return_keras_model:
        return model_in_using.model
    else:
        return model_in_using

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert4sqq.backend import keras, K, is_tf_keras
import tensorflow as tf


class Adam(keras.optimizers.Optimizer):
    """重新定义Adam优化器，便于派生出新的优化器
    （tensorflow的optimizer_v2类）
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        bias_correction=True,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'Adam'
        super(Adam, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = K.pow(beta_1_t, local_step)
        beta_2_t_power = K.pow(beta_2_t, local_step)

        # 更新公式
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * grad**2)
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(
                    v, indices, (1 - beta_2_t) * grad**2
                )

        # 返回算子
        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correction:
                m_t = m_t / (1.0 - beta_1_t_power)
                v_t = v_t / (1.0 - beta_2_t_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'bias_correction': self.bias_correction,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if is_tf_keras:
    print('is_tf_keras')
else:
    Adam = keras.optimizers.Adam

custom_objects = {
    'Adam': Adam
}

keras.utils.get_custom_objects().update(custom_objects)
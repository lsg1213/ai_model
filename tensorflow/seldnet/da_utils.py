import os
import numpy as np
import tensorflow as tf

EPSILON = 1e-8


'''
LABEL PRE-PROCESSING
'''
def degree_to_class(degrees,
                    resolution=20,
                    min_degree=0,
                    max_degree=180,
                    one_hot=True):
    degrees = np.array(degrees)
    n_classes = int((max_degree-min_degree)/resolution + 2)

    mask = np.logical_and(min_degree <= degrees, degrees <= max_degree)
    classes = mask * (degrees/resolution) + (1-mask) * (n_classes-1)
    classes = classes.astype(np.int32)

    if not one_hot:
        return classes 
    return np.eye(n_classes, dtype=np.float32)[classes]


def class_to_degree(classes,
                    resolution=20,
                    min_degree=0,
                    max_degree=180,
                    non_voice_value=-1):
    classes = np.array(classes)
    mask = classes != (max_degree-min_degree)/resolution + 1
    degrees = mask * (classes*resolution) + (1-mask) * non_voice_value
    return degrees


''' 
UTILS FOR FRAMES AND WINDOWS 
'''
def seq_to_windows(seq, 
                   window, 
                   skip=1,
                   padding=True, 
                   **kwargs):
    '''
    INPUT:
        seq: np.ndarray
        window: array of indices
            ex) [-3, -1, 0, 1, 3]
        skip: int
        padding: bool
        **kwargs: params for np.pad

    OUTPUT:
        windows: [n_windows, window_size, ...]
    '''
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window) + 1
    windows = window[np.newaxis, :] \
            + np.arange(0, len(seq), skip)[:, np.newaxis]
    if padding:
        seq = np.pad(
            seq,
            [[win_size//2, (win_size-1)//2]] + [[0, 0]]*len(seq.shape[1:]),
            **kwargs)

    return np.take(seq, windows, axis=0)


def windows_to_seq(windows,
                   window,
                   skip=1):
    '''
    INPUT:
        windows: np.ndarray (n_windows, window_size, ...)
        window: array of indices
        skip: int

    OUTPUT:
        seq
    '''
    n_window = windows.shape[0]
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window)

    seq_len = (n_window-1)*skip + 1
    seq = np.zeros([seq_len, *windows.shape[2:]], dtype=windows.dtype)
    count = np.zeros(seq_len)

    for i, w in enumerate(window):
        indices = np.arange(n_window)*skip - win_size//2 + w
        select = np.logical_and(0 <= indices, indices < seq_len)
        seq[indices[select]] += windows[select, i]
        count[indices[select]] += 1
    
    seq = seq / (count + EPSILON)
    return seq


'''
DATASET
'''
def window_generator(specs, 
                     labels, 
                     window_size, 
                     infinite=True):
    '''
    GENERATES WINDOWED SPECS AND LABELS

    INPUT:
        specs: continuous single spectrogram
        labels: continuous framewise labels
        window_size: 
        infinite: infinite generator or not
    OUTPUT:
        generator
    '''
    def generator():
        max_hop = window_size
        n_frames = len(specs)
        i = 0

        while True:
            i = (i + np.random.randint(1, max_hop+1)) % n_frames

            if i+window_size > n_frames:
                if not infinite:
                    break

                spec = np.concatenate(
                        [specs[i:], specs[:(i+window_size)%n_frames]],
                    axis=0)
                assert spec.shape[0] == window_size
            else:
                spec = specs[i:i+window_size]

            label = labels[(i+window_size//2) % n_frames] # center

            yield (spec, label)

    return generator


'''
MODEL
'''
def apply_kernel_regularizer(model, kernel_regularizer):
    model = tf.keras.models.clone_model(model)
    layer_types = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)
    for layer in model.layers:
        if isinstance(layer, layer_types):
            layer.kernel_regularizer = kernel_regularizer

    model = tf.keras.models.clone_model(model)
    return model


'''
OPTIMIZER
'''
class AdaBelief(tf.keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='AdaBelief',
                 **kwargs):
        super(AdaBelief, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBelief, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
                    (tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdaBelief, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = tf.compat.v1.assign(m, 
                               m * coefficients['beta_1_t'] + m_scaled_g_values,
                               use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * ((g_t-m_t) * (g_t-m_t))
        v = self.get_slot(var, 'v')
        grad_dev = grad - m_t 
        v_scaled_g_values = (grad_dev * grad_dev) * coefficients['one_minus_beta_2_t']
        v_t = tf.compat.v1.assign(v, 
                                  v * coefficients['beta_2_t'] + v_scaled_g_values,
                                  use_locking=self._use_locking)

        if not self.amsgrad:
            v_sqrt = tf.math.sqrt(v_t)
            var_update = tf.compat.v1.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = tf.math.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = tf.compat.v1.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = tf.math.sqrt(v_hat_t)
            var_update = tf.compat.v1.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t, v_hat_t])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = tf.compat.v1.assign(m, m * coefficients['beta_1_t'],
                                                     use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * ((g_t-m_t) * (g_t-m_t))
        v = self.get_slot(var, 'v')
        grad_dev = grad - m_t 
        v_scaled_g_values = (grad_dev * grad_dev) * coefficients['one_minus_beta_2_t']
        v_t = tf.compat.v1.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = tf.math.sqrt(v_t)
            var_update = tf.compat.v1.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = tf.math.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = tf.compat.v1.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = tf.math.sqrt(v_hat_t)
            var_update = tf.compat.v1.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return tf.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(AdaBelief, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config


def load_data(path, save=True):
    try:
        x = np.load(path + '_x.npy')
    except:
        from data_utils import from_wav_to_dataset
        x = from_wav_to_dataset(path, 'complex', pad=True) 
        if save:
            np.save(path + '_x.npy', x)

    try:
        y = np.load(path + '_y.npy')
    except:
        if not os.path.exists(path + '/angle.mat'):
            return x

        from scipy.io import loadmat
        y = loadmat(path + '/angle.mat')
        y = y['phi'].squeeze()
        if save:
            np.save(path + '_y.npy', y)

    assert len(x) == len(y)

    return x, y


class arg():
    gpus = '0'
    model = 'st_attention'
    pad_size = 19
    step_size = 9
    feature = 'mel'
    skip = 1
    dataset = 'noisex'
    norm = False
    noise_aug = False
    voice_aug = False
    aug = False
    snr = ['0']
    layer = -2
    algorithm = 'cam'
    class_index = 1
    before_softmax = -2
eager = True
config = arg()
if config.model == 'st_attention':
    config.before_softmax = -2
elif config.model == 'bdnn':
    config.before_softmax = -2

# test
# config.before_softmax = -1

from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

# eager = disable_eager_execution()

import glob, cv2
import numpy as np
import pickle
import scipy, os
import tensorflow as tf
import time
from tqdm import tqdm
print(tf.__version__)
from utils import preprocess_spec
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus


def sequence_to_windows(sequence, 
                        pad_size, 
                        step_size, 
                        skip=1,
                        padding=True, 
                        const_value=0):
    '''
    SEQUENCE: (time, ...)
    PAD_SIZE:  int -> width of the window // 2
    STEP_SIZE: int -> step size inside the window
    SKIP:      int -> skip windows...
        ex) if skip == 2, total number of windows will be halved.
    PADDING:   bool -> whether the sequence is padded or not
    CONST_VALUE: (int, float) -> value to fill in the padding

    RETURN: (time, window, ...)
    '''
    assert (pad_size-1) % step_size == 0
    window = np.concatenate([np.arange(-pad_size, -step_size, step_size),
                             np.array([-1, 0, 1]),
                             np.arange(step_size+1, pad_size+1, step_size)],
                            axis=0)
    window += pad_size
    output_len = len(sequence) if padding else len(sequence) - 2*pad_size
    window = window[np.newaxis, :] + np.arange(0, output_len, skip)[:, np.newaxis]

    if padding:
        pad = np.ones((pad_size, *sequence.shape[1:]), dtype=np.float32)
        pad *= const_value
        sequence = np.concatenate([pad, sequence, pad], axis=0)

    return np.take(sequence, window, axis=0)

def label_to_window(config, skip=1):
    def _preprocess_label(label):
        label = sequence_to_windows(
            label, config.pad_size, config.step_size, skip, True)
        return label
    return _preprocess_label



def windows_to_sequence(windows,
                        pad_size,
                        step_size):
    windows = np.array(windows)
    sequence = np.zeros((windows.shape[0],) + windows.shape[2:],
                        dtype=np.float32)   
    indices = np.arange(1, windows.shape[0]+1)
    indices = sequence_to_windows(
        indices, pad_size, step_size, True, -1)

    for i in range(windows.shape[0]):
        pred = windows[np.where(indices-1 == i)]
        sequence[i] = pred.mean(axis=0)
    
    return sequence

def image_resize(image, size=(7,80)):
    return tf.image.resize(image, size)

# @tf.function
def multipling(inputs):
    # import pdb; pdb.set_trace()
    conv, weights = inputs
    # weights = tf.expand_dims(weights,0)
    grad_cam = conv * weights
    grad_cam = tf.math.reduce_sum(grad_cam, axis=-1)
    return grad_cam

# @tf.function
def generate_grad_cam(model,data,class_idx,new_model):
    # data = (sound time, window, seq)
    img_tensor = tf.convert_to_tensor(data)

    # class index별 나눠서 진행하는 것 전처리
    class_templete = np.zeros(data.shape[0:2])
    masking = np.zeros(data.shape[1])
    ones = np.array([1])


    def get_grad_val(inputs):
        with tf.GradientTape() as y_tape:
            y_tape.watch(img_tensor)
            y_c = y_model(img_tensor, training=False)
        y_c_grad = y_tape.gradient(y_c, img_tensor)
        with tf.GradientTape() as A_tape:
            A_tape.watch(img_tensor)
            A_k = new_model(img_tensor, training=False)
        A_k_grad = A_tape.gradient(A_k, img_tensor)
        # import pdb; pdb.set_trace()
        # grad_val = tape.gradient(y_c, A_k)# gradient 찍기 argument 해보기 , y_c 랑 우측 arg
        return y_c_grad / A_k_grad, A_k
    
    # @tf.function
    def get_grad_val_window(inputs):
        y_c_grad = tf.zeros_like(inputs)
        for i in range(2**data.shape[1]):
            binary = bin(i)[2:]
            binary = '0' * (7 - len(binary)) + binary
            if int(binary[len(binary) // 2]) != class_idx:
                continue
            for j,k in enumerate(binary):
                if k == '1':
                    masking[j] = ones
            def y_mask(inputs):
                return inputs * masking
            # 이 부분에서 class_idx 다 반영해서 y_c_grad 값 뽑도록 수정
            with tf.GradientTape() as y_tape:
                y_tape.watch(img_tensor)
                y = y_model(img_tensor, training=False) 
                y_c = tf.map_fn(y_mask, y)
            y_c_grad += y_tape.gradient(y_c, img_tensor)
            
        y_c_grad /= 2 ** (data.shape[1] - 1) 
        # y_c_grad = tf.keras.utils.normalize(y_c_grad, axis=(1,2))
        # import pdb; pdb.set_trace()
        with tf.GradientTape() as A_tape:
            A_tape.watch(img_tensor)
            A_k = new_model(img_tensor, training=False)
            
        A_k_grad = A_tape.gradient(A_k, img_tensor)
        return y_c_grad / A_k_grad, A_k
    # import pdb;pdb.set_trace()
    grad_val, conv_output = get_grad_val_window(img_tensor)


    # import pdb; pdb.set_trace()
    if len(grad_val.shape) == 3:
        axis = (1,2)
    else:
        raise ValueError(f'grad_val shape is {grad_val.shape}')
    weights = tf.keras.backend.mean(tf.cast(grad_val, tf.float32), axis=axis)
    # 음성 길이, 모델 출력 channel
    for i in range(4-tf.rank(conv_output)):
        conv_output = conv_output[..., tf.newaxis]
    # conv_output = (time, 7, 10, 128), weights = (time,)
    cam = tf.map_fn(multipling, (conv_output, weights), dtype='float32')
    # cam = (time, 7, 10)
    
    cam = cam[..., tf.newaxis]
    cam = tf.map_fn(image_resize, cam)
    # cam = tf.image.resize(cam, (80, 7))
    ## Relu
    cam = tf.keras.activations.relu(cam)
    
    cam = tf.math.divide_no_nan(tf.squeeze(cam, -1), tf.keras.backend.max(cam,axis=-1))
 
    # return img_arr, cam, predictions
    return cam


def gradient_saliency(model, data):
    data = tf.convert_to_tensor(data)
    with tf.GradientTape() as tape:
        tape.watch(data)
        y = model(data)
    return tape.gradient(y, data).numpy()

if __name__ == '__main__':

    ## 2. image sources
    data_path = '/root/datasets/ai_challenge/TIMIT_noisex3'
    label_path = '/root/datasets/ai_challenge/TIMIT_noisex3'
    x = pickle.load(open(data_path + '/snr0_10.pickle', 'rb'))
    x = list(map(preprocess_spec(config, feature=config.feature), x))
    y = pickle.load(open(os.path.join(data_path, f'label_10.pickle'), 'rb'))
    y = list(map(label_to_window(config, skip=config.skip), y))
    H5_PATH = './TIMIT_noisex3_divideSNR/' \
        f'{config.model}_0.2_sgd_19_9_skip2_decay0.95_mel_batch4096_noiseaug_voiceaug_aug.h5'

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    model.summary()
    new_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=model.layers[config.layer].output)
    y_model = tf.keras.models.load_model(H5_PATH, compile=False)
    y_model.layers[config.before_softmax].activation = None
    class_idx = config.class_index
    x = x[:4]
    k = 0
    maps = []
    for s in tqdm(x):
        img = np.zeros((s.shape[0], s.shape[-1]))
        cam = np.zeros((s.shape[0], s.shape[-1]))
        tmp_cam = np.zeros_like(s)
        y_c = new_model(s, training=False)
        if config.algorithm == 'cam':
            ### grad-cam code ###
            _cam = generate_grad_cam(model, s, class_idx, new_model)
            # for i, j in enumerate(s):
            #     _cam = _generate_grad_cam(np.expand_dims(j, 0),model, class_idx, config.layer)
            #     tmp_cam[i] = np.expand_dims(_cam.T, 0)

            _img = s
            if _cam.shape[-1] != 80:
                _cam = tf.transpose(_cam, [0,2,1]  )
            if _img.shape[-1] != 80:
                _img = np.transpose(s, [0,2,1])

            cam = windows_to_sequence(_cam, config.pad_size, config.step_size)
            img = windows_to_sequence(_img, config.pad_size, config.step_size)
            #####################

        # print(np.array(cam).shape, np.array(img).shape)
        elif config.algorithm == 'sal':
            ### saliency code ###
            cam = gradient_saliency(model, s)
            cam = windows_to_sequence(cam, config.pad_size, config.step_size)
            img = windows_to_sequence(s, config.pad_size, config.step_size)
            #####################
        maps.append([img,cam])
    print('process done, save data')
    name = f'grad_{config.algorithm}_{config.model}_data_{config.snr[0]}_layer{config.layer}_class{config.class_index}'
    
    if config.algorithm == 'sal':
        name = name[5:]
    pickle.dump(maps, open(name, 'wb'))
    print(name)
    print(model.layers[config.layer].name)
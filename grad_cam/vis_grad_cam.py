class arg():
    gpus = '0'
    model = 'bdnn'
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
    layer = -3
    algorithm = 'cam'

config = arg()

import glob
import numpy as np
import pickle
import scipy, os
import tensorflow as tf
import time
from tqdm import tqdm

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
def generate_grad_cam(model,data,class_idx,new_model):
    # data = (sound time, frame time, seq)
    img_tensor = tf.convert_to_tensor(data)
    # import pdb; pdb.set_trace()
    
    conv_output = new_model.layers[-1].trainable_variables
    # import pdb; pdb.set_trace()
    conv_output = conv_output[0]

    @tf.function
    def get_grad_val(inputs):
        with tf.GradientTape() as tape:
            y_c = new_model(img_tensor, training=False)
        grad_val = tape.gradient(y_c, new_model.layers[-1].trainable_variables)[0]
        return grad_val
    grad_val = tf.map_fn(get_grad_val, img_tensor)
    
    # weights = tf.keras.backend.mean(grad_val, axis=(1))
    # weights = tf.expand_dims(weights, -1)
    weights = tf.keras.backend.mean(grad_val, axis=(1))

    cam = tf.math.multiply(tf.expand_dims(conv_output,0), tf.expand_dims(weights,1))
    cam = tf.cast(cam, tf.float32)
    cam = cam[..., tf.newaxis]
    # cam = tf.map_fn(image_resize, cam)
    cam = tf.image.resize(cam, (80, 7))
    ## Relu
    cam = tf.keras.activations.relu(cam)
    
    cam = tf.math.divide_no_nan(cam, tf.keras.backend.max(cam))
    cam = tf.squeeze(cam, -1)
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
    data_path = '/root/datasets/ai_challenge/TIMIT_noisex3/snr0_10.pickle'
    x = pickle.load(open(data_path, 'rb'))
    x = list(map(preprocess_spec(config, feature=config.feature), x))

    H5_PATH = './TIMIT_noisex3_divideSNR/' \
        'bdnn_0.1_sgd_19_9_skip2_decay0.95_mel_batch4096_noiseaug_voiceaug_aug.h5'

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    model.summary()
    new_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=model.layers[config.layer].output)
    class_idx = 1
    x = x[:len(x) // 100]
    percent = len(x) // 100
    k = 0
    maps = []
    for s in tqdm(x):
        img = cam = np.zeros((s.shape[0], s.shape[-1]))
        if config.algorithm == 'cam':
            ### grad-cam code ###
            _cam = generate_grad_cam(model, s, class_idx, new_model)
            _img = s
            if _cam.shape[-1] != 80:
                _cam = tf.transpose(_cam, [0,2,1])
            if _img.shape[-1] != 80:
                _img = np.transpose(s, [0,2,1])

            cam = windows_to_sequence(_cam, config.pad_size, config.step_size)
            img = windows_to_sequence(_img, config.pad_size, config.step_size)
            # import pdb; pdb.set_trace()
                
            #####################

        # print(np.array(cam).shape, np.array(img).shape)
        # import pdb; pdb.set_trace()
        elif config.algorithm == 'sal':
            ### saliency code ###
            cam = gradient_saliency(model, s)
            cam = windows_to_sequence(cam, config.pad_size, config.step_size)
            img = windows_to_sequence(s, config.pad_size, config.step_size)
            #####################
        maps.append([img,cam])
        
    print('process done, save data')    
    pickle.dump(maps, open(f'grad_{config.algorithm}_data_{config.snr[0]}_layer{config.layer}', 'wb'))
    print(f'grad_cam_data_{config.snr[0]}_layer{config.layer}')

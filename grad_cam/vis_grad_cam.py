class arg():
    gpus = '-1'
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
    layer = 21
    algorithm = 'cam'
eager = True
config = arg()
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

@tf.function
def multipling(inputs):
    conv, weights = inputs
    # weights = tf.expand_dims(weights,0)
    weights = weights[tf.newaxis, tf.newaxis, ...]
    grad_cam = conv * weights
    grad_cam = tf.math.reduce_sum(grad_cam, axis=-1)
    return grad_cam


# @tf.function
def generate_grad_cam(model,data,class_idx,new_model):
    # data = (sound time, window, seq)
    img_tensor = tf.convert_to_tensor(data)
    inp = model.input
    y_c = model.output.op.inputs[0][0, class_idx]
    A_k = model.get_layer('tf_op_layer_mul_3').output
    # A_k = model.layers[config.layer].output
    
    # _conv_output = new_model.layers[-1].trainable_variables
    # conv_output = _conv_output[0]    # [1] is bias
    # get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    # [conv_output, grad_val, model_output] = get_output([img_tensor])
    
    # @tf.function
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

    grad_val, conv_output = get_grad_val(img_tensor)

    # A_k = model.layers[activation_layer].output
    
    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.


    # weights = tf.keras.backend.mean(grad_val, axis=(1))
    # weights = tf.expand_dims(weights, -1)
    weights = tf.keras.backend.mean(tf.cast(grad_val, tf.float32), axis=(1,2))
    # 음성 길이, 모델 출력 channel

    # conv_output = (time, 7, 10, 128), weights = (time, 128)
    cam = tf.map_fn(multipling, (conv_output, weights), dtype='float32')
    # cam = (time, 7, 10)
    
    cam = cam[..., tf.newaxis]
    cam = tf.map_fn(image_resize, cam)
    # import pdb; pdb.set_trace()
    # cam = tf.image.resize(cam, (80, 7))
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


import tensorflow.keras.backend as K
K.set_learning_phase(False)
def _generate_grad_cam(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
    model: pretrained resnet50 모델 (include_top=True)
    class_index: 이미지넷 정답 레이블
    activation_layer: 시각화하려는 레이어 이름

    return:
    grad_cam: grad_cam 히트맵
    """
    inp = model.input
    y_c = model.output.op.inputs[0][0, class_index]
    A_k = model.get_layer('tf_op_layer_mul_3').output
    # A_k = model.layers[activation_layer].output
    
    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])

    ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
    # import pdb; pdb.set_trace()
    conv_output = conv_output[0]
    grad_val = grad_val[0]
    
    ## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
    weights = np.mean(grad_val, axis=(0, 1))
    
    ## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]
        
        import pdb; pdb.set_trace()
    
    grad_cam = cv2.resize(grad_cam, (7, 80))
    

    ## ReLU를 씌워 음수를 0으로 만든다.
    grad_cam = np.maximum(grad_cam, 0)

    grad_cam = grad_cam / grad_cam.max()
    return grad_cam

if __name__ == '__main__':
    

    ## 2. image sources
    data_path = '/root/datasets/ai_challenge/TIMIT_noisex3/snr0_10.pickle'
    x = pickle.load(open(data_path, 'rb'))
    x = list(map(preprocess_spec(config, feature=config.feature), x))

    H5_PATH = './TIMIT_noisex3_divideSNR/' \
        f'{config.model}_0.2_sgd_19_9_skip2_decay0.95_mel_batch4096_noiseaug_voiceaug_aug.h5'

    model = tf.keras.models.load_model(H5_PATH, compile=False)
    model.summary()
    new_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=model.layers[config.layer].output)
    y_model =  tf.keras.models.load_model(H5_PATH, compile=False)
    y_model.layers[-2].activation = None
    class_idx = 1
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
    name = f'grad_{config.algorithm}_{config.model}_data_{config.snr[0]}_layer{config.layer}'
    pickle.dump(maps, open(name, 'wb'))
    print(name)

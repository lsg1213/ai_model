import tensorflow as tf
import numpy as np
import time, os
from models import resnet_v1
from utils import *
from sklearn.metrics import confusion_matrix
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# baseline - [0.3897, 0.213, 0.3811, 0.3977, 0.1912, 0.2639, 0.224, 0.1873, 0.352] - 0.28888

# sample의 weight을 diag(confusion_matrix)로 하였다.
# CLASS_WISE - [0.4124, 0.4065, 0.3928, 0.3985, 0.378]
# max(0.5, l_cnt/(l_cnt+(len(u) / 2))))
# - [0.3914, 0.3699, 0.3925, 0.3879]
# max(0.5, l_cnt/(l_cnt+(len(u) / 2)))) + Version2
# - [0.3785, 0.4083, 0.3915, 0.4166, 0.3783]
# max(1/3, l_cnt/(l_cnt+(len(u) / 2))))
# - [0.443, 0.3903, 0.3899, 0.4158, 0.4269]
# max(0.25, l_cnt/(l_cnt+len(u)))
# - [0.4089, 0.392, 0.3838, 0.4082, 0.405, 0.4113]

# max(0.25, l_cnt/(l_cnt+len(u))) + Version2
# (u_hat_label == i) * (u_hat[:, i] >= np.percentile(u_hat[:, i][u_hat_label == i], 1-ratio))
# - [0.4465, 0.3851, 0.4106, 0.4007, 0.445, 0.4002, 0.4284] - 0.41664
# max(0.25, l_cnt/(l_cnt+len(u))) + Version2-Fixed
# (u_hat_label == i) * (u_hat[:, i] >= np.percentile(u_hat[:, i][u_hat_label == i], (1-ratio)*100))
# - [0.4086, 0.4092, 0.4248, 0.4011, 0.4009, 0.4223] - 0.41115
# Version2 - Fixed + np.where(u_hat > 0.1, u_hat - 0.1, 0)
# - [0.4354, 0.3872, 0.3947, 0.431, 0.4218, 0.4037] - 0.4123
# Version2 - Fixed + Softmax
# - [0.3985, 0.4117, 0.4107, 0.3987, 0.4229, 0.3425, 0.4198] - 0.40069
# Version2 - Fixed + Tau=0.5
# - [0.3915, 0.3816, 0.3858, 0.3848, 0.3908] - 0.3869
# Version2 - Fixed + Tau=0.2
# - [0.4265, 0.4255, 0.414, 0.4045, 0.3902, 0.3912] - 0.40865
# Version2 - Fixed + Tau=0.1
# - [0.442, 0.4417, 0.4417, 0.409, 0.4163, 0.4043, 0.4064, 0.4176, 0.3925, 0.4427, 0.3722, 0.393] - 0.41495
# Version2 - Fixed + Tau=0.1 + without filtering
# - [0.3633, 0.37]
# Version2 - Fixed + Tau=0.1 + max(1/2, l_cnt/(l_cnt+len(u)))
# - [0.3779, 0.4267, 0.3968, 0.3954, 0.4216] - 0.40368
# Version2 - Fixed + Tau=0.1 + max(1/4, l_cnt/(l_cnt+len(u)))
# - [0.4156, 0.4264, 0.4163, 0.4268, 0.4078] - 0.41858
# Version2 - Fixed + Tau=0.1 + max(1/8, l_cnt/(l_cnt+len(u)))
# - [0.4154, 0.4012, 0.4008, 0.422, 0.363] - 0.40048
# Version2 - Fixed + Tau=0.1 + max(1/16, l_cnt/(l_cnt+len(u)))
# - [0.3442, 0.4078, 0.387, 0.4093, 0.3961] - 0.3888
# Version2 - Fixed + Tau=0.1 + MT(0.9)
# - [0.408, 0.4152, 0.4204, 0.2135, 0.4054, 0.2356, 0.4094]
# Version2 - Fixed + Tau=0.05
# - [0.4094, 0.3949, 0.4228, 0.3945, 0.3905, 0.4117] -
# Version2 - Fixed - one-hot
# - [0.4402, 0.392, 0.4191, 0.4116, 0.3975, 0.4099, 0.4026] - 0.41041
# max(0.25, l_cnt/(l_cnt+len(u))) + Version2-Fixed - no GAMMA
# - [0.381, 0.37, 0.3591, 0.3759, 0.3459] - 0.36638

# - Version2 - Fixed + using only training...
# [0.3892, 0.4104, 0.41, 0.3927, 0.4008, 0.401]
# - Version2 - using 500 validation
# [0.4201, 0.4117, 0.4102, 0.4026, 0.4162, 0.4082]
# - weights norm
# [0.4167, 0.3966,
# Version2 - using 500 validation + separate loss
# [0.4427, 0.3884, 0.3992, 0.4043, 0.4347, 0.447]

# tau = best_acc     : [0.4091,0.4243, 0.3909, 0.3755, 0.4394] - 0.40784
# tau = best_acc / 2 : [0.4157, 0.422, 0.4021, 0.4378, 0.4318] - 0.42188
# tau = best_acc / 4 : [0.4078, 0.3917, 0.427, 0.3808, 0.4152] - 0.40450
# tau = 0.4 - best_acc / 2 : [0.4004, 0.4141, 0.3974]
# tau = 0.2 - best_acc / 4 : [0.4017,

# Version3 - [0.4366, 0.4158, 0.4106, 0.408, 0.4216, 0.4102, 0.3689, 0.406] : 0.40971
# Version3 - no balanced sampling
# - [0.3586, 0.434, 0.4143, 0.3807, 0.3984, 0.4173, 0.4018, 0.4091] - 0.40178
# Version3 - u_hat[:, i] <= np.percentile(u_hat[:, i][u_hat_label == i], ratio*100)
# - [0.399, 0.1848, 0.2143, 0.3843, 0.3602]

# WHOLE - [0.3759, 0.3892, 0.413]
# max(0.25, l_cnt/(l_cnt+len(u)))
# - [0.3675, 0.3983, 0.3935, 0.4072, 0.4146, 0.3741]


# per-class
# 1 - [0.3922, 0.409, 0.3903, 0.3912]
# 1 - [0.3658, 0.3488, 0.3934, 0.3802]
# 2 - [0.3864, 0.413, 0.3837, 0.4021]
# 5 - [0.4029, 0.404, 0.3988, 0.4164]
# 5+norm - [0.4192, 0.3943, 0.3905, 0.3832]
# train 줄여~
# 1 - [0.3771, 0.349, 0.3901, 0.3529]
# 1+norm - [0.369, 0.3791, 0.3511, 0.3434]
# no gamma - [0.3865, 0.3846, 0.4119, 0.3829, 0.3981]
# 5+norm - [0.379, 0.4056, 0.3811, 0.4009, 0.3909]
# 5 - [0.3713, 0.368, 0.3498, 0.3864, 0.3747]

# training set
# [0.3988, 0.3698]
# 510 [0.2858, 0.4063, 0.3875,
# 550 [
np.set_printoptions(precision=3)


@tf.function
def predict(model, x, batch_size=32):
    y = []
    for i in range((len(x)-1)//batch_size + 1):
        y.append(model(x[i*batch_size:(i+1)*batch_size], training=False))
    return tf.concat(y, axis=0)


def fit(model,
        optimizer,
        labeled,
        unlabeled,
        val_data,
        epochs,
        batch_size,
        model_name):
    # HYPER-PARAMETERS
    EARLY_STOP = 32
    GAMMA = 0

    # init
    # random_choices = np.reshape(
    #     [np.random.choice(np.arange(len(labeled[1]))[labeled[1].argmax(axis=-1) == i], size=5, replace=False)
    #     for i in range(10)],
    #     (-1,))
    random_choices = np.reshape(
        [np.random.choice(np.arange(len(val_data[1]))[np.argmax(val_data[1], axis=-1) == i], size=5, replace=False)
        for i in range(10)],
        (-1,))

    gamma_val_x = np.take(val_data[0], random_choices, axis=0)
    gamma_val_y = np.take(val_data[1], random_choices, axis=0).argmax(axis=-1)

    # rest = [(i not in random_choices) for i in range(len(labeled[0]))]
    # rest = np.array(rest, dtype=np.bool)
    # labeled = (labeled[0][rest], labeled[1][rest])

    l_cnt = len(labeled[0]) # labeled data count...
    u_cnt = len(unlabeled)
    l_queue = tf.queue.RandomShuffleQueue(
        l_cnt, 0, ['float32', 'int32'], shapes=[labeled[0].shape[1:], labeled[1].shape[1:]])
    l_queue.enqueue_many(labeled)

    labeler = tf.keras.models.clone_model(model)

    best_acc = 0.
    best_count = 0
    train_loss = tf.keras.metrics.Mean()

    val_x = val_data[0]
    val_y = val_data[1].numpy().argmax(axis=-1)

    def stage_two(epoch, best_acc):
        return True # best_acc > 0.2

    for epoch in range(epochs):
        t = time.time()

        # pseudo labeling
        if stage_two(epoch, best_acc):
            if best_count ==  0:
                GAMMA = np.diag(confusion_matrix(
                    labeled[1].argmax(axis=-1), # gamma_val_y,
                    predict(labeler, labeled[0][:, 4:-4, 4:-4], batch_size=batch_size).numpy().argmax(axis=-1),
                    normalize='pred'))
                # print(GAMMA)
                # modify GAMMA
                # GAMMA = GAMMA / (np.mean(GAMMA) + 1e-2)
                # GAMMA = GAMMA * 0 + 1
                print(GAMMA)

                u_hat = labeler.predict(unlabeled, batch_size=batch_size)
                # u_hat = np.eye(10)[u_hat.argmax(axis=-1)] # softmax to one-hot
                # u_hat = np.where(u_hat > 0.2, u_hat - 0.1, 0)
                # u_hat /= u_hat.sum(axis=-1, keepdims=True) + 1e-8
                tau = max(0.2 - best_acc/4, 1e-4) # 0.1 # dynamic tau
                u_hat = np.log(u_hat / u_hat.max(axis=-1, keepdims=True) +1e-8) # softmax to logits
                u_hat = np.exp(u_hat / tau)
                u_hat = u_hat / u_hat.sum(axis=-1, keepdims=True) # logits to softmax

                # Class-wise...
                # u_hat_label = u_hat.argmax(axis=-1)
                # label_counts = np.eye(10)[u_hat_label].sum(axis=0)
                # select = u_hat_label * 0
                # size = label_counts.min()
                # for i in range(10):
                #     ratio = size / (label_counts[i] + 1e-8)
                #     select += (u_hat_label == i) * (np.random.random((u_cnt,)) < ratio)
                # select = select.astype(np.bool)

                # Class-wise... version 2 fixed (balancing...)
                u_hat_label = u_hat.argmax(axis=-1)
                label_counts = np.eye(10)[u_hat_label].sum(axis=0)
                select = u_hat_label * 0
                size = np.percentile(label_counts, 25)
                if size > 0:
                    for i in range(10):
                        if label_counts[i] <= 0:
                            continue
                        ratio = size / (label_counts[i] + 1e-8)
                        ratio = min(max(0, ratio), 1)
                        mask = u_hat[:, i] >= np.percentile(u_hat[:, i][u_hat_label == i], (1-ratio)*100)
                        mask = mask * (np.random.random((u_cnt,)) < size / mask.sum())
                        select += (u_hat_label == i) * mask
                select = select.astype(np.bool)

                # Class-wise... version 3 (added s_rate)
                # s_rate = confusion_matrix(
                #     val_y,
                #     predict(labeler, val_x, batch_size=batch_size).numpy().argmax(axis=-1),
                #     normalize='true').sum(axis=0)
                # s_rate = 1 / (s_rate + 1e-1)
                # u_hat_label = u_hat.argmax(axis=-1)
                # label_counts = np.eye(10)[u_hat_label].sum(axis=0)
                # select = u_hat_label * 0
                # size = np.percentile(label_counts, 25)
                # if size > 0:
                #     for i in range(10):
                #         if label_counts[i] <= 0:
                #             continue
                #         ratio = s_rate[i] * size / (label_counts[i] + 1e-8)
                #         ratio = min(max(0, ratio), 1)
                #         mask = u_hat[:, i] >= np.percentile(u_hat[:, i][u_hat_label == i], (1-ratio)*100)
                #         mask = mask * (np.random.random((u_cnt,)) < size / mask.sum())
                #         select += (u_hat_label == i) * mask
                # select = select.astype(np.bool)

                # Whole Dataset
                # s_rate = confusion_matrix(
                #     val_y,
                #     predict(labeler, val_x, batch_size=batch_size).numpy().argmax(axis=-1),
                #     normalize='true').sum(axis=0)
                # s_rate = 1 / (s_rate + 1e-1)
                # label_counts = np.eye(10)[u_hat.argmax(axis=-1)].sum(axis=0)
                # size = np.percentile(label_counts, 25)
                # count = 0
                # if size > 0:
                #     for i in range(10):
                #         count += np.clip(s_rate[i] * size, 0, label_counts[i])
                #
                # confidence = u_hat.max(axis=-1)
                # select = confidence >= np.percentile(confidence, 100*(1-count/u_cnt))
                # select = np.logical_and(select, np.random.random((u_cnt,)) < count / select.sum())
                # select = confidence >= np.percentile(confidence, 100-int(best_acc*50))

                print(np.sum(select))

                u = unlabeled[select]
                u_hat = u_hat[select]

            # shuffle
            if len(u) > 0:
                indices = np.random.permutation(len(u))
                u = u[indices]
                u_hat = u_hat[indices]
            import pdb; pdb.set_trace()

            l_size = int(batch_size * max(1/4, l_cnt/(l_cnt+len(u))))
            u_size = batch_size - l_size
        else:
            l_size = batch_size

        # train
        for i in range(l_cnt//l_size):
            mini_batch = l_queue.dequeue_many(l_size)
            x, y = mini_batch

            if stage_two(epoch, best_acc) and len(u) > 0:
                pseudo_x, pseudo_y = u[u_size*i:u_size*(i+1)], u_hat[u_size*i:u_size*(i+1)]

                x = tf.concat([x, pad(pseudo_x)], axis=0)
                y = tf.concat([y, pseudo_y], axis=0)
                weights = tf.concat(
                    [tf.ones(l_size), np.dot(pseudo_y, GAMMA)], axis=0)
                is_labeled = tf.concat([tf.ones(l_size), tf.zeros(len(pseudo_y))], axis=0)
            else:
                weights = tf.ones(BATCH_SIZE)
                is_labeled = tf.ones(BATCH_SIZE)
            is_labeled = tf.expand_dims(is_labeled, axis=-1)

            # Augmentation
            x_cnt = len(x) # BATCH_SIZE
            crop = np.random.randint(2, size=(x_cnt, 1, 1, 1)).astype(np.float32)
            x = tf.image.random_crop(x, [x_cnt, 32, 32, 3]) * crop \
                + tf.image.random_crop(x, [x_cnt, 32, 32, 3]) * (1 - crop)
            reverse = np.random.randint(2, size=(x_cnt, 1, 1, 1)).astype(np.float32)
            x = x * reverse + tf.reverse(x, axis=(2,)) * (1 - reverse)

            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                y_hat = model(x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y, y_hat)
                # y = tf.cast(y, tf.float32)
                # y_mul_coef = is_labeled*y + (1-is_labeled)*y*GAMMA[None, :]
                # loss = -tf.math.reduce_sum(
                #     tf.cast(y_mul_coef, tf.float32)*tf.math.log(y_hat), axis=-1)
                loss = tf.math.reduce_mean(loss * weights)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            l_queue.enqueue_many(mini_batch)
            train_loss.update_state(loss)

        print(f'{epoch}:: TRAIN LOSS {train_loss.result().numpy():.4f}', end=' ')
        train_loss.reset_states()

        # validate
        val_y_hat = predict(model, val_x, batch_size=batch_size)
        acc = tf.math.reduce_mean(
            tf.keras.metrics.categorical_accuracy(val_data[1], val_y_hat))
        loss = tf.math.reduce_mean(
            tf.keras.losses.categorical_crossentropy(val_data[1], val_y_hat))
        print(f"({time.time()-t:.3f}) ACC: {acc:.3f}, LOSS: {loss:.3f}")

        # save
        if acc > best_acc:
            print('updated from {:.4f} to {:.4f}'.format(best_acc, acc))
            model.save(model_name)
            best_acc = acc
            best_count = 0
            labeler.set_weights(model.get_weights()) # update labeler
        else:
            best_count += 1
            if best_count >= EARLY_STOP:
                print('early stopping')
                break
        # mean_average(labeler, model)
        # For Mean Teacher
        # if mean_teacher:
        #     mean_average(labeler, model)
        # else:
        #     if np.percentile(
        #         np.diag(confusion_matrix(val_y, val_y_hat.numpy().argmax(axis=-1), normalize='pred')),
        #         25) > 0:
        #         mean_teacher = True
        #         print("Mean Teacher On~")
        #         mean_average(labeler, model)


if __name__ == '__main__':
    # hyper-parameters
    BATCH_SIZE = 128
    EPOCHS = 10000
    MODEL_NAME = 'C:\\Users\\Daniel\\Documents\\test_.h5'

    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # one-hot to vector
    print(x_train.shape, x_test.shape)
    y_train = np.eye(10)[np.squeeze(y_train)]
    y_test = np.eye(10)[np.squeeze(y_test)]

    TRAIN_SIZE = 45000
    indices = np.arange(len(x_train))
    indices = np.random.permutation(indices)
    x_train, x_val = x_train[indices[:TRAIN_SIZE]], x_train[indices[TRAIN_SIZE:]]
    y_train, y_val = y_train[indices[:TRAIN_SIZE]], y_train[indices[TRAIN_SIZE:]]

    x, y = x_train[:550], y_train[:550]
    x = pad(x)
    u = x_train[5000:]
    # y_true = np.argmax(y_train[5000:], axis=-1)

    # Model
    model = resnet_v1([32, 32, 3], 20, num_classes=10)
    opt = tf.keras.optimizers.Adam() # SGD(0.01, momentum=0.9)

    print(np.unique(np.argmax(y, axis=-1), return_counts=True))

    fit(model,
        opt,
        (x, y),
        u,
        (tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME)

    model.load_weights(MODEL_NAME)
    print(tf.math.reduce_mean(
        tf.keras.metrics.categorical_accuracy(
            y_test, model.predict(x_test, batch_size=BATCH_SIZE))))

import argparse, os, tqdm, yaml, pdb
import tensorflow as tf
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
from mixmatch import mixmatch, semi_loss, linear_rampup, interleave, weight_decay, ema
from model import WideResNet
from preprocess import fetch_dataset, _list_to_tf_dataset

def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=None, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')

    parser.add_argument('--epochs', type=int, default=1024, help='number of epochs, (default: 1024)')
    parser.add_argument('--batch-size',  type=int, default=64, help='examples per batch (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning_rate, (default: 0.01)')

    parser.add_argument('--labelled-examples', type=int, default=500, help='number labelled examples (default: 4000')
    parser.add_argument('--validation-examples', type=int, default=5000, help='number validation examples (default: 5000')
    parser.add_argument('--val-iteration', type=int, default=1024, help='number of iterations before validation (default: 1024)')
    parser.add_argument('--T', type=float, default=0.5, help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=3, help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda-u', type=int, default=100, help='multiplier for unlabelled loss (default: 100)')
    parser.add_argument('--rampup-length', type=int, default=16,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='ema decay for ema model vars (default: 0.999)')

    # parser.add_argument('--config-path', type=str, default=None, help='path to yaml config file, overwrites config')
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard visualization')
    parser.add_argument('--resume', action='store_true', help='whether to restore from previous training runs')
    parser.add_argument('--gpus', type=str, default='0', help='set gpu numbers')
    parser.add_argument('--conf_num', type=int, default=10, help='confusion matrix dataset number')

    return parser.parse_args()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    start_epoch = 0
    log_path = f'.logs/{args.dataset}@{args.labelled_examples}@idea'
    ckpt_dir = f'{log_path}/checkpoints'

    # tfds version 2.1.0
    datasetX, datasetU, val_dataset, test_dataset, confusion_dataset, num_classes = fetch_dataset(args, log_path)
    batching = lambda dataset: dataset.batch(batch_size=len(confusion_dataset))
    confusion_batch = next(iter(batching(confusion_dataset)))
    
    model = WideResNet(num_classes, depth=28, width=2)
    model.build(input_shape=(None, 32, 32, 3))

    labeler = WideResNet(num_classes, depth=28, width=2)
    labeler.build(input_shape=(None, 32, 32, 3))
    labeler.set_weights(model.get_weights())

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    model_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(model_ckpt, f'{ckpt_dir}/model', max_to_keep=3)

    ema_model = WideResNet(num_classes, depth=28, width=2)
    ema_model.build(input_shape=(None, 32, 32, 3))
    ema_model.set_weights(model.get_weights())
    ema_ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=ema_model)
    ema_manager = tf.train.CheckpointManager(ema_ckpt, f'{ckpt_dir}/ema', max_to_keep=3)

    if args.resume:
        model_ckpt.restore(manager.latest_checkpoint)
        ema_ckpt.restore(manager.latest_checkpoint)
        model_ckpt.step.assign_add(1)
        ema_ckpt.step.assign_add(1)
        start_epoch = int(model_ckpt.step)
        print(f'Restored @ epoch {start_epoch} from {manager.latest_checkpoint} and {ema_manager.latest_checkpoint}')

    train_writer = None
    if args.tensorboard:
        train_writer = tf.summary.create_file_writer(f'{log_path}/train')
        val_writer = tf.summary.create_file_writer(f'{log_path}/validation')
        test_writer = tf.summary.create_file_writer(f'{log_path}/test')


    # assigning args used in functions wrapped with tf.function to tf.constant/tf.Variable to avoid memory leaks
    args.T = tf.constant(args.T)
    args.beta = tf.Variable(0., shape=())
    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        # 아이디어 추가
        data = confusion_batch['image']
        label = confusion_batch['label']

        GAMMA = np.diag(confusion_matrix(
            np.argmax(label.numpy(), -1), # gamma_val_y,
            np.argmax(labeler(data, training=False).numpy(), -1),
            normalize='pred')
            )
 
        xe_loss, l2u_loss, total_loss, accuracy = train(datasetX, datasetU, model, ema_model, labeler, GAMMA, optimizer, epoch, args, best_acc)
        val_xe_loss, val_accuracy = validate(val_dataset, ema_model, epoch, args, split='Validation')
        test_xe_loss, test_accuracy = validate(test_dataset, ema_model, epoch, args, split='Test')
        if best_acc < val_accuracy:
            best_acc = val_accuracy
        if (epoch - start_epoch) % 16 == 0:
            model_save_path = manager.save(checkpoint_number=int(model_ckpt.step))
            ema_save_path = ema_manager.save(checkpoint_number=int(ema_ckpt.step))
            print(f'Saved model checkpoint for epoch {int(model_ckpt.step)} @ {model_save_path}')
            print(f'Saved ema checkpoint for epoch {int(ema_ckpt.step)} @ {ema_save_path}')

        model_ckpt.step.assign_add(1)
        ema_ckpt.step.assign_add(1)

        step = args.val_iteration * (epoch + 1)
        if args.tensorboard:
            with train_writer.as_default():
                tf.summary.scalar('xe_loss', xe_loss.result(), step=step)
                tf.summary.scalar('l2u_loss', l2u_loss.result(), step=step)
                tf.summary.scalar('total_loss', total_loss.result(), step=step)
                tf.summary.scalar('accuracy', accuracy.result(), step=step)
            with val_writer.as_default():
                tf.summary.scalar('xe_loss', val_xe_loss.result(), step=step)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=step)
            with test_writer.as_default():
                tf.summary.scalar('xe_loss', test_xe_loss.result(), step=step)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=step)


    if args.tensorboard:
        for writer in [train_writer, val_writer, test_writer]:
            writer.flush()

# @tf.function
# def predict(model, x, batch_size=32):
#     y = []
#     for i in range((len(x)-1)//batch_size + 1):
#         y.append(model(x[i*batch_size:(i+1)*batch_size], training=False))
#     return tf.concat(y, axis=0)

def train(datasetX, datasetU, model, ema_model, labeler, GAMMA, optimizer, epoch, args, best_acc):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args.batch_size, drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(datasetX))
    iteratorU = iter(shuffle_and_batch(datasetU))
    # unlabeled = tf.convert_to_tensor([data['image'] for data in datasetU])

    # u_hat = labeler.predict(unlabeled, batch_size=args.batch_size) # 원래 logit 뱉음

    # tau = tf.convert_to_tensor(max(0.2 - best_acc/4, 1e-4))

    # u_hat = tf.math.exp(u_hat / tau)
    # u_hat = u_hat / tf.reduce_sum(u_hat, -1, True)
    # u_hat_label = tf.argmax(u_hat,-1)

    # label_counts = tf.reduce_sum(np.eye(10)[u_hat_label], 0, True)
    # select = u_hat_label * 0
    # size = np.percentile(label_counts, 25)
    # if size > 0:
    #     for i in range(u_hat.shape[-1]):
    #         if label_counts[i] <= 0:
    #             continue
    #         ratio = size / (label_counts[i] + tf.convert_to_tensor(1e-8))
    #         ratio = min(max(0, ratio), 1)
    #         mask = u_hat[:, i] >= np.percentile(u_hat[:, i][u_hat_label == i])
    #         select += (u_hat_label == i) * mask
    # select = tf.cast(select, tf.bool)
    # pseudo_dataset = unlabeled[select]
    # u_hat = u_hat[select]
    # if len(u) > 0:
    #     indices = np.random.permutation(len(pseudo_dataset))
    #     pseudo_dataset = pseudo_dataset[indices]
    #     u_hat = u_hat[indices]

    # pseudo_dataset = _list_to_tf_dataset([{'image': pseudo_dataset[indice],
    #                     'label': u_hat[indice] for indice in range(len(pseudo_dataset))])
    # iteratorP = iter(shuffle_and_batch(pseudo_dataset))

    progress_bar = tqdm.tqdm(range(args.val_iteration), unit='batch')
    for batch_num in progress_bar:
        lambda_u = args.lambda_u * linear_rampup(epoch + batch_num/args.val_iteration, args.rampup_length)
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(datasetX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            batchU = next(iteratorU)

        args.beta.assign(np.random.beta(args.alpha, args.alpha))
        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, batchX['image'], batchX['label'], batchU['image'], args.T, args.K, args.beta)
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, args.batch_size)
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)
            weights = tf.convert_to_tensor([np.dot(logits_u, GAMMA)])
            # compute loss
            xe_loss, l2u_loss = semi_loss(XUy[:args.batch_size], logits_x, XUy[args.batch_size:], logits_u, tf.cast(weights, tf.double))
            total_loss = xe_loss + lambda_u * l2u_loss

        
        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema(model, ema_model, args.ema_decay)
        weight_decay(model=model, decay_rate=args.weight_decay * args.learning_rate)

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32), training=False))

        progress_bar.set_postfix({
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy

if __name__ == "__main__":
    main(get_args())
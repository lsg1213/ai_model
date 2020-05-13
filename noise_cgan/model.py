import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, Sequential
import keras.backend as K
from random import randint
import matplotlib.pyplot as plt
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EPOCHS = 100
noise_dim = 100
BATCH_SIZE = 256
class_num = 10

def generator(output_shape=(28,28,1), class_num=class_num, stddev=0.2, z_dim=noise_dim):
    model = Sequential()

    model.add(Dense(256, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))

    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(class_num, z_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)

    



def discriminator(input_shape=(28,28,1), class_num=class_num, stddev=0.2):
    noise_input = Input(shape=input_shape)
    reshaped_noise = Flatten()(noise_input)

    model = Sequential()

    model.add(Dense(512, input_dim=np.prod(input_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=input_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(class_num, np.prod(input_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)
    
    
    return Model(inputs=[noise_input, label_input], outputs=out)






if __name__ == "__main__":
        
    generator = generator()
    discriminator = discriminator()
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # def discriminator_loss(real_output, fake_output):
    #     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #     total_loss = real_loss + fake_loss
    #     return total_loss

    # def generator_loss(fake_output):
    #     return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(0.0002*5, 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    discriminator.compile(loss=['binary_crossentropy'],
            optimizer=discriminator_optimizer,
            metrics=['acc'])
    noise = Input(shape=(noise_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])
    discriminator.trainable = False
    valid = discriminator([img, label])
    combined = Model([noise, label], valid)
    combined.compile(loss=['binary_crossentropy'],
            optimizer=generator_optimizer)
            
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    (train_noises, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


    def generate_and_save_noises(model, epoch, test_input):
        # `training`이 False로 맞춰진 것을 주목하세요.
        # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
        label = np.random.rand(test_input.shape[0],1) * 10 // 1
        label = label.astype('int')
        predictions = model((test_input, tf.reshape(label,(test_input.shape[0], -1))), training=False)
        

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('noises/noise_at_epoch_{:04d}.png'.format(epoch))

    def sample_noises(generator, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, noise_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = generator.predict([noise, sampled_labels])
        # Rescale noises 0 ~ 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("noises/%d.png" % epoch)
        plt.close()

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            d_loss_real = [0,0]
            d_loss_fake = [0,0]
            g_loss = 0
            step = 0
            
            for noise_batch, label_batch in dataset:
                noise_batch = tf.expand_dims(noise_batch, axis=-1)
                noise_batch = tf.cast(noise_batch, tf.float32)

                noise = np.random.normal(0, 1, (noise_batch.shape[0], noise_dim))
                # labels = tf.one_hot(labels,10)
                label_batch = tf.reshape(label_batch,(noise_batch.shape[0], -1))
                generated_noise = generator.predict([noise,label_batch])

                valid = np.ones((noise_batch.shape[0], 1))
                fake = np.zeros((noise_batch.shape[0], 1))

                d_loss_real = np.add(d_loss_real, discriminator.train_on_batch([noise_batch, label_batch], valid))
                d_loss_fake = np.add(d_loss_fake, discriminator.train_on_batch([generated_noise, label_batch], fake))
                sampled_labels = np.random.randint(0, class_num, noise_batch.shape[0])
                g_loss = np.add(g_loss, combined.train_on_batch([noise, sampled_labels], valid))
                step += 1
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0]/step, 100*d_loss[1]/step, g_loss/step))
            

            # GIF를 위한 이미지를 바로 생성합니다.
            # generate_and_save_noises(generator,
            #                             epoch + 1,
            #                             seed)
            sample_noises(generator, epoch+1)

            # 5 에포크가 지날 때마다 모델을 저장합니다.
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        sample_noises(generator, epochs)

    train_noises = (train_noises - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.
    BUFFER_SIZE = 60000
    perm = np.random.permutation(train_noises.shape[0])
    train_noises = train_noises[perm]
    train_labels = train_labels[perm]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_noises, train_labels)).batch(BATCH_SIZE)
    
    train(train_dataset, EPOCHS)
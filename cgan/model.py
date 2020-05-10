
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
    # noise_input = Input(shape=(z_dim,))
    # label_input = Input(shape=(1,))
    # label_embedding = Embedding(class_num, z_dim)(label_input)
    # label_embedding = Flatten()(label_embedding)

    # model_input = Multiply()([noise_input, label_embedding])
    
    # z = Dense(1024,kernel_initializer=tf.random_normal_initializer(stddev=stddev))(model_input)
    # z = BatchNormalization()(z)
    # z = ReLU()(z)
    # z = Dense(128*7*7,kernel_initializer=tf.random_normal_initializer(stddev=stddev))(z)
    # z = BatchNormalization()(z)
    # z = ReLU()(z)
    # z = Reshape([7,7,128])(z)
    # z = Conv2DTranspose(64, (4,4), (2,2),padding='same',kernel_initializer=tf.random_normal_initializer(stddev=stddev))(z)
    # z = BatchNormalization()(z)
    # z = ReLU()(z)
    # out = Conv2DTranspose(1, (4,4), (2,2),padding='same',kernel_initializer=tf.random_normal_initializer(stddev=stddev))(z)
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

    # model.summary()

    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(class_num, z_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)
    return Model(inputs=[noise_input, label_input], outputs=out)

    



def discriminator(input_shape=(28,28,1), class_num=class_num, stddev=0.2):
    image_input = Input(shape=input_shape)
    reshaped_image = Flatten()(image_input)

    # label_input = Input(shape=(1,))
    # label_embedding = Embedding(class_num, np.prod(input_shape))(label_input)
    # label_embedding = Flatten()(label_embedding)
    # y = Multiply()([reshaped_image, label_embedding])
    
    # y = Reshape(input_shape)(y)
    # z = Conv2D(64,(4,4),(2,2),padding='same',kernel_initializer=tf.random_normal_initializer(stddev=stddev))(y)
    # z = ReLU(negative_slope=0.2)(z)
    # z = Conv2D(128,(4,4),(2,2),padding='same',kernel_initializer=tf.random_normal_initializer(stddev=stddev))(z)
    # z = BatchNormalization()(z)
    # z = ReLU(negative_slope=0.2)(z)
    # z = Flatten()(z)
    # z = Dense(1024,kernel_initializer=tf.random_normal_initializer(stddev=stddev))(z)
    # z = BatchNormalization()(z)
    # logit = ReLU(negative_slope=0.2)(z)
    # out = Dense(1,activation='sigmoid',kernel_initializer=tf.random_normal_initializer(stddev=stddev))(logit)
    
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
    
    
    return Model(inputs=[image_input, label_input], outputs=out)






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
    
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # 이 시드를 시간이 지나도 재활용하겠습니다. 
    # (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.) 
    # seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    # `tf.function`이 어떻게 사용되는지 주목해 주세요.
    # 이 데코레이터는 함수를 "컴파일"합니다.
    # @tf.function
    # def train_step(images, labels, epoch):
        # noise = tf.random.normal([images.shape[0], noise_dim])
        
    
        # tf.print (f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

        
        # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #     generated_images = generator([noise, labels], training=True)
            
        #     real_output = discriminator([tf.reshape(images,(images.shape[0],28,28,1)), labels], training=True)
        #     fake_output = discriminator([generated_images, labels], training=True)
        #     gen_loss = generator_loss(fake_output)
        #     disc_loss = discriminator_loss(real_output, fake_output)
        #     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        #     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def generate_and_save_images(model, epoch, test_input):
        # `training`이 False로 맞춰진 것을 주목하세요.
        # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
        label = np.random.rand(test_input.shape[0],1) * 10 // 1
        label = label.astype('int')
        predictions = model((test_input, tf.reshape(label,(test_input.shape[0], -1))), training=False)
        

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))

    def sample_images(generator, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, noise_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = generator.predict([noise, sampled_labels])
        # Rescale images 0 ~ 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            d_loss_real = [0,0]
            d_loss_fake = [0,0]
            g_loss = 0
            step = 0
            
            for image_batch, label_batch in dataset:
                image_batch = tf.expand_dims(image_batch, axis=-1)
                image_batch = tf.cast(image_batch, tf.float32)

                noise = np.random.normal(0, 1, (image_batch.shape[0], noise_dim))
                # labels = tf.one_hot(labels,10)
                label_batch = tf.reshape(label_batch,(image_batch.shape[0], -1))
                generated_image = generator.predict([noise,label_batch])

                valid = np.ones((image_batch.shape[0], 1))
                fake = np.zeros((image_batch.shape[0], 1))

                d_loss_real = np.add(d_loss_real, discriminator.train_on_batch([image_batch, label_batch], valid))
                d_loss_fake = np.add(d_loss_fake, discriminator.train_on_batch([generated_image, label_batch], fake))
                sampled_labels = np.random.randint(0, class_num, image_batch.shape[0])
                g_loss = np.add(g_loss, combined.train_on_batch([noise, sampled_labels], valid))
                step += 1
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0]/step, 100*d_loss[1]/step, g_loss/step))
            

            # GIF를 위한 이미지를 바로 생성합니다.
            # generate_and_save_images(generator,
            #                             epoch + 1,
            #                             seed)
            sample_images(generator, epoch+1)

            # 15 에포크가 지날 때마다 모델을 저장합니다.
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # 마지막 에포크가 끝난 후 생성합니다.
        # generate_and_save_images(generator,
        #                         epochs,
        #                         seed)
        sample_images(generator, epochs)

    train_images = (train_images - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.
    BUFFER_SIZE = 60000
    perm = np.random.permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(BATCH_SIZE)
    
    train(train_dataset, EPOCHS)
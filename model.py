import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers.spectral_normalization import SpectralNormalization


class GAN(keras.Model):
    def __init__(self, gex_size, num_cells_generate):
        super(GAN, self).__init__()
        self.hyperparams = self.Hyperparamaters(gex_size=gex_size, num_cells_generate=num_cells_generate)
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()

    class Hyperparamaters(object):
        def __init__(self, gex_size, num_cells_generate):
            self.num_cells_generate = num_cells_generate
            self.gex_size = gex_size
            self.num_epochs = 1000
            self.batch_size = 2048 # todo
            self.latent_dim = 128
            self.gen_layer_1 = 256
            self.gen_layer_2 = 512
            self.gen_layer_3 = 1024
            self.disc_layer_1 = 1024
            self.disc_layer_2 = 512
            self.disc_layer_3 = 256
            self.gen_learn_rate = 0.0001
            self.disc_learn_rate = 0.0004
            self.disc_train_steps = 1
            self.adam_b1 = 0.50
            self.adam_b2 = 0.999

    def compile(self, **kwargs):
        super(GAN, self).compile()
        self.d_optimizer = Adam(learning_rate=self.hyperparams.disc_learn_rate,
                                beta_1=self.hyperparams.adam_b1,
                                beta_2=self.hyperparams.adam_b2)
        self.g_optimizer = Adam(learning_rate=self.hyperparams.gen_learn_rate,
                                beta_1=self.hyperparams.adam_b1,
                                beta_2=self.hyperparams.adam_b2)
        self.d_loss = self.get_discriminator_loss
        self.g_loss = self.get_generator_loss

    # model architecture
    def get_generator(self):
        gen = keras.Sequential([
            SpectralNormalization(layers.Dense(
                input_shape=(self.hyperparams.latent_dim,),
                units=self.hyperparams.gen_layer_1,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=self.hyperparams.gen_layer_2,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=self.hyperparams.gen_layer_3,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=self.hyperparams.gex_size,
                activation=None))])
        return gen

    def get_discriminator(self):
        disc = keras.Sequential([
            SpectralNormalization(layers.Dense(
                input_shape=(self.hyperparams.gex_size,),
                units=self.hyperparams.disc_layer_1,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=self.hyperparams.disc_layer_2,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=self.hyperparams.disc_layer_3,
                activation="selu",
                kernel_initializer="lecun_normal")),
            SpectralNormalization(layers.Dense(
                units=1,
                activation=None))])
        return disc

    # relativistic average hinge loss
    def get_discriminator_loss(self, real, fake):
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

        return real_loss + fake_loss

    def get_generator_loss(self, real, fake):
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 - fake_logit))
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 + real_logit))

        return real_loss + fake_loss

    # sample latent space
    def sample_latent(self, batch_size):
        return tf.random.normal(shape=(batch_size, self.hyperparams.latent_dim))

    # req for subclassed keras models
    def call(self, inputs, **kwargs):
        return self.generator(inputs)

    # end-to-end training
    def train_step(self, real):
        batch_size = tf.shape(real)[0]

        # Train the discriminator
        for i in range(self.hyperparams.disc_train_steps):
            # Sample the latent space
            random_latent_vectors = self.sample_latent(batch_size)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vectors
                fake = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real, training=True)
                # Calculate the discriminator loss using the fake and real image logits
                d_loss = self.d_loss(real=real_logits, fake=fake_logits)

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        random_latent_vectors = self.sample_latent(batch_size)
        with tf.GradientTape() as tape:
            # Sample the latent space
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            real_img_logits = self.discriminator(real, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss(real=real_img_logits, fake=gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

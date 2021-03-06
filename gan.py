import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers.spectral_normalization import SpectralNormalization


class GAN(keras.Model):
    class Hyperparamaters(object):
        def __init__(self, gex_size, num_cells_generate):
            self.num_cells_generate = num_cells_generate
            self.gex_size = gex_size
            self.num_epochs = 1000
            self.batch_size = 2048
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

    def __init__(self, gex_size, num_cells_generate, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.hyperparams = self.Hyperparamaters(gex_size=gex_size, num_cells_generate=num_cells_generate)
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()

    def compile(self, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.d_optimizer = Adam(learning_rate=self.hyperparams.disc_learn_rate,
                                beta_1=self.hyperparams.adam_b1,
                                beta_2=self.hyperparams.adam_b2)
        self.g_optimizer = Adam(learning_rate=self.hyperparams.gen_learn_rate,
                                beta_1=self.hyperparams.adam_b1,
                                beta_2=self.hyperparams.adam_b2)
        self.d_loss = self.get_discriminator_loss
        self.g_loss = self.get_generator_loss

    # req for subclassed keras models, see: model.build
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 128), dtype=tf.dtypes.float32)])
    def call(self, inputs):
        return self.discriminator(self.generator(inputs))

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

    # relativistic average hinge losses
    def get_discriminator_loss(self, real, fake):
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

        return tf.nn.compute_average_loss(real_loss + fake_loss, global_batch_size=self.hyperparams.batch_size * tf.distribute.get_strategy().num_replicas_in_sync)

    def get_generator_loss(self, real, fake):
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 - fake_logit))
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 + real_logit))

        return tf.nn.compute_average_loss(real_loss + fake_loss, global_batch_size=self.hyperparams.batch_size * tf.distribute.get_strategy().num_replicas_in_sync)

    # sample latent space, a 3D gaussian hypersphere
    def sample_latent(self, batch_size):
        return tf.random.normal(shape=(batch_size, self.hyperparams.latent_dim))

    @tf.function
    def train_step(self, real):
        per_replica_losses = tf.distribute.get_strategy().run(self._train_step, args=(real,))
        return tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # end-to-end training
    def _train_step(self, real):
        batch_size = tf.shape(real)[0]

        # train the discriminator
        for i in range(self.hyperparams.disc_train_steps):
            # sample the latent space
            latent_vectors = self.sample_latent(batch_size)
            with tf.GradientTape() as tape:
                # generate fakes from the latent vectors
                fake = self.generator(latent_vectors, training=True)
                # get logits for reals & fakes
                real_logits = self.discriminator(real, training=True)
                fake_logits = self.discriminator(fake, training=True)
                # calculate the discriminator loss using the real and logits
                d_loss = self.d_loss(real=real_logits, fake=fake_logits)

            # get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # train the generator
        latent_vectors = self.sample_latent(batch_size)
        with tf.GradientTape() as tape:
            # sample latent space
            gen_cells = self.generator(latent_vectors, training=True)
            # get discriminator logits for reals & fakes
            gen_cell_logits = self.discriminator(gen_cells, training=True)
            real_cell_logits = self.discriminator(real, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss(real=real_cell_logits, fake=gen_cell_logits)

        # get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        # summary stats
        fake_batch = self.generator(self.sample_latent(batch_size))
        d_real_acc = tf.reduce_mean(tf.sigmoid(self.discriminator(real)))
        d_fake_acc = tf.reduce_mean(tf.sigmoid(self.discriminator(fake_batch)))
        d_total_acc = tf.reduce_mean((d_real_acc, d_fake_acc))
        real_mean = tf.reduce_mean(real)
        fake_mean = tf.reduce_mean(fake_batch)

        return {"d_loss": d_loss,
                "g_loss": g_loss,
                "d_real_acc": d_real_acc,
                "d_fake_acc": d_fake_acc,
                "d_total_acc": d_total_acc,
                "real_mean": real_mean,
                "fake_mean": fake_mean}

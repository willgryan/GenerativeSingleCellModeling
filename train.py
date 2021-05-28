import datetime

import scanpy as sc
import tensorflow as tf
from tensorflow import keras

from gan import GAN
import utils

class TrainingMetrics(keras.callbacks.Callback):
    def __init__(self, tb_callback, **kwargs):
        super(TrainingMetrics, self).__init__(**kwargs)
        self.tb_callback = tb_callback

    def on_epoch_end(self, epoch):
        if (epoch % 100 == 0):  # every 100 epochs
            print ("") #todo mmd

        if (epoch % 10 == 0): #every 10 epochs
            random_latent_vectors = self.model.sample_latent(batch_size=self.model.hyperparams.batch_size)
            generated_cells = self.model.generator(random_latent_vectors)
            real_cells = test_tf.batch(batch_size=model.hyperparams.batch_size, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator().next()

            w = self.tb_callback._train_writer # use the same summary writer as the tensorboard callback
            with w.as_default():
                tf.summary.histogram(data=generated_cells, step=epoch, name="generated_cells")
                tf.summary.histogram(data=real_cells, step=epoch, name="real_cells")
                w.flush()

if __name__ == "__main__":
    utils.set_seeds() # soft-determinism for now

    tb = True # using tensorboard in pycharm
    if(tb):
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', utils.LOG_DIR])
        url = tb.launch()
        print(url)

    # initial scanpy train & test splits
    data = sc.read_h5ad('data/GSE144136_preprocessed.h5ad')
    train = sc.pp.subsample(data=data, fraction=0.90, copy=True, random_state=utils.RANDOM)
    test = sc.pp.subsample(data=data, fraction=0.10, copy=True, random_state=utils.RANDOM)

    # build model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = GAN(gex_size=train.shape[1], num_cells_generate=test.shape[0])
        model.compile()
        model.build(input_shape=(model.hyperparams.batch_size, model.hyperparams.latent_dim))  # req. for subclassed models

    # process data for training
    train_tf = tf.data.Dataset.from_tensor_slices(train.X). \
        cache(). \
        shuffle(buffer_size=train.shape[0], seed=utils.RANDOM). \
        batch(batch_size=model.hyperparams.batch_size * strategy.num_replicas_in_sync, num_parallel_calls=tf.data.AUTOTUNE). \
        prefetch(buffer_size=tf.data.AUTOTUNE)
    train_tf_distributed = strategy.experimental_distribute_dataset(train_tf)

    test_tf = tf.data.Dataset.from_tensor_slices(test.X). \
        cache(). \
        shuffle(buffer_size=test.shape[0], seed=utils.RANDOM). \
        prefetch(buffer_size=tf.data.AUTOTUNE)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=utils.LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                 update_freq='epoch',
                                                 write_graph=False,
                                                 profile_batch=0)

    model.fit(x=train_tf_distributed,
              epochs=model.hyperparams.num_epochs,
              steps_per_epoch=int (train.shape[0] / model.hyperparams.batch_size), # steps = # batches per epoch
              callbacks=[tb_callback,
                         TrainingMetrics(tb_callback),
                         tf.keras.callbacks.ModelCheckpoint(period=100, filepath='training/model_{epoch}')])
    # Save model
    model.save("training/model_final")

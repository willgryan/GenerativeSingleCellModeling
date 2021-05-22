import scanpy as sc
import tensorflow as tf

from model import GAN
import utils

if __name__ == "__main__":
    utils.set_seeds() # soft-determinism for now

    # initial scanpy train & test splits
    data = sc.read_h5ad('data/GSE144136_preprocessed.h5ad')
    train = sc.pp.subsample(data=data, fraction=0.90, copy=True, random_state=utils.RANDOM)
    test = sc.pp.subsample(data=data, fraction=0.10, copy=True, random_state=utils.RANDOM)

    # build model
    model = GAN(gex_size=train.shape[1], num_cells_generate=test.shape[0])
    model.compile()
    model.build(input_shape=(model.hyperparams.batch_size, model.hyperparams.latent_dim))  # req. for subclassed models

    # process data for training
    train_tf = tf.data.Dataset.from_tensor_slices(train.X). \
        cache(). \
        shuffle(buffer_size=train.shape[0], seed=utils.RANDOM). \
        batch(batch_size=model.hyperparams.batch_size, num_parallel_calls=tf.data.AUTOTUNE). \
        prefetch(buffer_size=tf.data.AUTOTUNE)

    model.fit(train_tf, epochs=model.hyperparams.num_epochs)

    # Save model
    model(model.sample_latent(1))  # todo - subclassing errors
    model.save("trained_model")
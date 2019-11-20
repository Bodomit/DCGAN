import os
import datetime
import time
import argparse
import glob

from functools import partial

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import DCGAN

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Script arguments.
parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--dataset-dir', '-d', metavar='DIR',
                    help='Dataset root path.')
parser.add_argument('--results-dir', '-r', metavar='DIR', default="~/results", 
                    help="Results root path.")
parser.add_argument('--result-subdir', '-s', metavar='DIR', default=None, 
                    help="Path to specific experiment results.")
parser.add_argument('--data-fname-pattern', '-p', metavar='PATTERN', default='*.jpg')
parser.add_argument('--batch-size', '-b', metavar='N', default=64)
parser.add_argument('--iterations', '-n', metavar='N', default=60000)

def _escape_directory(directory: str) -> str:
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    return directory

def _result_subdirectory(timestamp_format : str = "%Y%m%d.%H%M%S") -> str:
    t = time.time()
    ts = datetime.datetime.fromtimestamp(t).strftime(timestamp_format)
    return "{}_{}".format(ts, "dcgan")

def _result_dir(results_dir, result_subdir) -> str:
    if result_subdir is None:
        result_subdir = _result_subdirectory()
    result_dir = os.path.join(results_dir, result_subdir)
    return _escape_directory(result_dir)

def _get_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_png(image, channels=3)

    image = tf.image.resize(image, [64, 64])
    image = tf.image.random_flip_left_right(image)

    image = (image / 127.5) - 1

    return image

def _get_data(dataset_dir : str, data_fname_pattern: str):
    # Get the file names.
    data_path = os.path.join(dataset_dir, data_fname_pattern)
    data = list(glob.glob(data_path))
    
    # Convert to TF dataset.
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(len(data))
    dataset = dataset.map(_get_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()

    return dataset

def _validate(generator, validation_set, validation_directory, gen_i, n_rows=4, n_cols=10):
        path = os.path.join(
            validation_directory,
            "{:09d}.svg".format(gen_i+1))
        images = generator(validation_set, training=False)

        images = images + 1
        images = images * 127.5
        images = tf.dtypes.cast(images, tf.uint8)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 8))
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i, j].imshow(images[(i*n_cols)+j],
                                norm=matplotlib.colors.Normalize())
                axs[i, j].axis("off")
        fig.savefig(path)
        plt.close('all')
        del axs
        del fig


def main(dataset_dir: str, 
         data_fname_pattern: str, 
         out_dir: str,
         batch_size: int,
         iterations: int):

    # Setup sub-directories.
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoint_dir)
    validation_dir = os.path.join(out_dir, "validation")
    os.makedirs(validation_dir)

    # Get data.
    dataset = _get_data(dataset_dir, data_fname_pattern)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    # Get the GAN.
    dcgan = DCGAN(checkpoint_dir, batch_size=batch_size)

    # Get validation set and function.
    validation_set = tf.random.normal([40, dcgan.noise_vec_length])
    validate = partial(_validate, dcgan.generator, validation_set, validation_dir)
    
    # Training
    print("Starting Training")
    validate(-1)
    
    with tqdm(desc="Iteration: ",
              total=iterations,
              dynamic_ncols=True) as pbar:
        i_generator_loss_avg = tf.keras.metrics.Mean()
        i_discriminator_loss_avg = tf.keras.metrics.Mean()

        for i, batch in enumerate(dataset):

            losses = dcgan.train_step(batch)
            pbar.update()

            for loss in losses["generator_loss"]:
                i_generator_loss_avg(loss)
            for loss in losses["discriminator_loss"]:
                i_discriminator_loss_avg(loss)

            pbar.set_postfix(g_loss=i_generator_loss_avg.result().numpy(),
                                 d_loss=i_discriminator_loss_avg.result().numpy())

            if (i + 1) % 100 == 0:
                validate(i)
                dcgan.checkpoint_manager.save()
                i_generator_loss_avg.reset_states()
                i_discriminator_loss_avg.reset_states()

    # Save the final model.
    dcgan.save(os.path.join(out_dir, "resulting_model"))

if __name__ == "__main__":
    args = parser.parse_args()

    result_dir = _result_dir(args.results_dir, args.result_subdir)
    dataset_dir = _escape_directory(args.dataset_dir)
    data_fname_pattern = args.data_fname_pattern

    main(dataset_dir,
         data_fname_pattern, 
         result_dir,
         args.batch_size,
         args.iterations)



import os
import tensorflow as tf

class DCGAN:

    def __init__(self,
                 checkpoint_path: str,
                 batch_size: int = 64,
                 noise_vec_length: int = 100):
        self.batch_size = batch_size
        self.noise_vec_length = noise_vec_length

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()

        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, checkpoint_path, max_to_keep=5)


    @staticmethod
    def discriminator_model() -> tf.keras.Model:

        def conv2d(filters) -> tf.keras.layers.Conv2D:
            init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
            conv = tf.keras.layers.Conv2D(
                filters, 5, strides=(2, 2), 
                padding='same',
                kernel_initializer=init)
            return conv

        model = tf.keras.Sequential([
            conv2d(64*1),
            tf.keras.layers.LeakyReLU(),

            conv2d(64*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            conv2d(64*4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            conv2d(64*8),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])
        return model


    @staticmethod
    def generator_model() -> tf.keras.Model:

        def conv2d_transpose(filters) -> tf.keras.layers.Conv2DTranspose:
            init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
            conv_t = tf.keras.layers.Conv2DTranspose(
                filters, 5, strides=(2, 2), 
                kernel_initializer=init,
                padding='same')
            return conv_t

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64*8*4*4),

            tf.keras.layers.Reshape((4, 4, 64*8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            conv2d_transpose(64*4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            conv2d_transpose(64*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),

            conv2d_transpose(64*1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            
            conv2d_transpose(3),
            tf.keras.layers.Activation("tanh")
        ])
        return model

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(real_output), real_output)

        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_vec_length])
        
        disc_loss = self._train_disc(images, noise)
        gen1_loss = self._train_gen(noise)
        gen2_loss = self._train_gen(noise)

        return {
            "discriminator_loss": [disc_loss],
            "generator_loss": [gen1_loss, gen2_loss]
        }


    @tf.function
    def _train_gen(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss


    @tf.function
    def _train_disc(self, images, noise):
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss

    def save(self, path: str):
        tf.saved_model.save(self.generator, os.path.join(path, "gen/1"))
        tf.saved_model.save(self.discriminator, os.path.join(path, "disc/1"))
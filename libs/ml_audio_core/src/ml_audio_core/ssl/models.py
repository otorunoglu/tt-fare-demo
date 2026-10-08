import tensorflow as tf
import numpy as np
from typing import Any


class SimCLR(tf.keras.Model):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
    Adapted for Audio.
    """

    def __init__(
        self,
        base_encoder: tf.keras.Model,
        projection_dim: int = 128,
        temperature: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            base_encoder: The base model (e.g. CNN) that outputs the representation h.
            projection_dim: Dimensionality of the projection head output z.
            temperature: Temperature parameter for NT-Xent loss.
        """
        super().__init__(**kwargs)
        self.base_encoder = base_encoder
        self.temperature = temperature

        # Projection Head (MLP with one hidden layer)
        # Assuming base_encoder output dim is known or inferred (e.g. 2048)
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(projection_dim, activation="relu"),
                tf.keras.layers.Dense(projection_dim),  # Linear output layer for z
            ],
            name="projection_head",
        )

    def call(self, inputs):
        """
        Forward pass for inference/training.

        Args:
            inputs: tuple (view1, view2) or single tensor.

        Returns:
            If training (tuple): (z1, z2) - projections
            If inference (single): h - representation
        """
        # Handle dual input for training
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            x1, x2 = inputs

            # Encoder -> Representation h
            h1 = self.base_encoder(x1)
            h2 = self.base_encoder(x2)

            # Projector -> Projection z
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            return z1, z2

        else:
            # Single input (inference mode) -> return representation
            return self.base_encoder(inputs)

    def compute_loss(self, z1, z2):
        """Compute NT-Xent loss."""
        # Normalize projections
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        batch_size = tf.shape(z1)[0]

        # Cosine similarity matrix
        # (2N, 2N)
        z = tf.concat([z1, z2], axis=0)
        sim_matrix = tf.matmul(z, z, transpose_b=True)

        # Mask out self-similarity
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        # Set self-similarity to very small number to be ignored in softmax
        sim_matrix = tf.where(mask, -1e9, sim_matrix)

        # Scale by temperature
        logits = sim_matrix / self.temperature

        # Labels: i-th sample in z1 matches i-th sample in z2
        # z = [z1_0 ... z1_N, z2_0 ... z2_N]
        # target for z1_i is z2_i (index N+i)
        # target for z2_j is z1_j (index j-N)

        labels_1 = tf.range(batch_size) + batch_size
        labels_2 = tf.range(batch_size)
        labels = tf.concat([labels_1, labels_2], axis=0)

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        return tf.reduce_mean(loss)

    def train_step(self, data):
        """Custom training step."""
        # Unpack data (X, y) - y is dummy
        if isinstance(data, (tuple, list)):
            X, _ = data
        else:
            X = data

        with tf.GradientTape() as tape:
            z1, z2 = self(X, training=True)
            loss = self.compute_loss(z1, z2)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}


def create_default_encoder(input_shape=(64, 96, 1), output_dim=128):
    """
    Create a simple 2D CNN encoder.
    Adjust input_shape to match your AudioProcessor settings (n_mels, fixed_tbins, channels).
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Output representation h
    x = tf.keras.layers.Dense(output_dim, activation="relu")(x)

    return tf.keras.Model(inputs, x, name="default_encoder")

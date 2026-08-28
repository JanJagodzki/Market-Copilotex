import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


def build_mlp(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),

            layers.Dense(
                256,
                activation="relu",
            ),
            layers.Dropout(0.3),

            layers.Dense(
                128,
                activation="relu",
            ),
            layers.Dropout(0.2),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return compile_model(model)


def build_cnn(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv1D(
                64,
                3,
                activation="relu",
                padding="causal",
            ),

            layers.Conv1D(
                64,
                3,
                activation="relu",
                padding="causal",
            ),

            layers.GlobalAveragePooling1D(),

            layers.Dense(
                64,
                activation="relu",
            ),

            layers.Dropout(0.2),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return compile_model(model)


def build_lstm(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.LSTM(
                64,
                return_sequences=True,
            ),

            layers.LSTM(32),

            layers.Dropout(0.2),

            layers.Dense(
                32,
                activation="relu",
            ),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return compile_model(model)


def build_gru(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.GRU(
                64,
                return_sequences=True,
            ),

            layers.GRU(32),

            layers.Dropout(0.2),

            layers.Dense(
                32,
                activation="relu",
            ),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return compile_model(model)


def build_tcn(input_shape):
    inputs = keras.Input(
        shape=input_shape
    )

    x = inputs

    for dilation in [1, 2, 4, 8]:
        residual = x

        x = layers.Conv1D(
            64,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
        )(x)

        x = layers.Dropout(0.15)(x)

        if residual.shape[-1] != 64:
            residual = layers.Conv1D(
                64,
                kernel_size=1,
            )(residual)

        x = layers.Add()(
            [x, residual]
        )

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
    )

    return compile_model(model)


def build_cnn_lstm(input_shape):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv1D(
                64,
                3,
                padding="causal",
                activation="relu",
            ),

            layers.MaxPooling1D(2),

            layers.LSTM(
                64,
                return_sequences=True,
            ),

            layers.LSTM(32),

            layers.Dropout(0.2),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    return compile_model(model)


def transformer_block(
    x,
    head_size,
    num_heads,
    ff_dim,
    dropout,
):
    attention = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
    )(x, x)

    x = layers.Add()(
        [x, attention]
    )

    x = layers.LayerNormalization()(x)

    feed_forward = layers.Dense(
        ff_dim,
        activation="relu",
    )(x)

    feed_forward = layers.Dense(
        x.shape[-1]
    )(feed_forward)

    x = layers.Add()(
        [x, feed_forward]
    )

    return layers.LayerNormalization()(x)


def build_transformer(input_shape):
    inputs = keras.Input(
        shape=input_shape
    )

    x = layers.Dense(64)(inputs)

    for _ in range(3):
        x = transformer_block(
            x,
            head_size=16,
            num_heads=4,
            ff_dim=128,
            dropout=0.1,
        )

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
    )

    return compile_model(model)


def get_neural_models(input_shape):
    return {
        "MLP": build_mlp(input_shape),
        "1D CNN": build_cnn(input_shape),
        "LSTM": build_lstm(input_shape),
        "GRU": build_gru(input_shape),
        "TCN": build_tcn(input_shape),
        "CNN-LSTM": build_cnn_lstm(input_shape),
        "Transformer": build_transformer(
            input_shape
        ),
    }
def build_patchtst(input_shape):
    patch_size = 5
    embed_dim = 64
    number_of_patches = input_shape[0] // patch_size

    inputs = tf.keras.Input(
        shape=input_shape
    )

    x = tf.keras.layers.Conv1D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)

    positions = tf.range(
        start=0,
        limit=number_of_patches,
        delta=1,
    )

    position_embedding = (
        tf.keras.layers.Embedding(
            input_dim=number_of_patches,
            output_dim=embed_dim,
        )(positions)
    )

    x = x + position_embedding

    for _ in range(2):
        attention = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=16,
                dropout=0.1,
            )(
                x,
                x,
            )
        )

        x = tf.keras.layers.Add()(
            [
                x,
                attention,
            ]
        )

        x = (
            tf.keras.layers.LayerNormalization()
            (x)
        )

        feed_forward = (
            tf.keras.layers.Dense(
                128,
                activation="gelu",
            )(x)
        )

        feed_forward = (
            tf.keras.layers.Dropout(
                0.1
            )(feed_forward)
        )

        feed_forward = (
            tf.keras.layers.Dense(
                embed_dim
            )(feed_forward)
        )

        x = tf.keras.layers.Add()(
            [
                x,
                feed_forward,
            ]
        )

        x = (
            tf.keras.layers.LayerNormalization()
            (x)
        )

    x = (
        tf.keras.layers.GlobalAveragePooling1D()
        (x)
    )

    x = tf.keras.layers.Dense(
        64,
        activation="gelu",
    )(x)

    x = tf.keras.layers.Dropout(
        0.2
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PatchTST",
    )

    return compile_model(model)

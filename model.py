import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

def create_model(number_of_classes, weights_path=None):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(None, 2)))
    # model.add(layers.LSTM(64, input_shape=(None, 2), return_sequences = True))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(number_of_classes))

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=0.01),
        metrics=["sparse_categorical_accuracy"],
    )

    if weights_path != None:
        model.load_weights(weights_path)

    return model




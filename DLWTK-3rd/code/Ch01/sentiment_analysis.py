import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

max_len = 200
n_words = 10_000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE = 500


def load_data():
    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)
    # We also have a convenient way of padding sentences to max_len, so that we can use all sentences,
    # whether short or long, as inputs to a neural network with
    # an input vector of fixed size
    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
    return (X_train, y_train), (X_test, y_test)


def build_model():
    model = models.Sequential()
    # For now, let’s assume that the embedding() layer will map the sparse space of words
    # contained in the reviews into a denser space.
    # 嵌入层
    # The model will take as input an integer matrix of size (batch, input_length).
    # The model will output dimension (input_length, dim_embedding).
    # The largest integer in the input should be no larger
    # than n_words (vocabulary size).
    model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(layers.Dropout(.3))

    # Takes the maximum value of either feature vector from each of the n_words features.
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


(X_train, y_train), (X_test, y_test) = load_data()
model = build_model()
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
score = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("\nTest score:", score[0])
print("Test accuracy:", score[1])

# Predicting output
# predictions = model.predict(X)
# For a given input, several types of output can be computed including a method model.evaluate() used
# to compute the loss values, a method model.predict_classes() used to compute category outputs,
# and a method model.predict_proba() used to compute class probabilities.

"""
@Description: Testing different optimizers in TensorFlow
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-31 10:31:23
"""

from tensorflow import keras
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = .2  # how much train is reserved for validation
DROPOUT = .3
RESHAPE = 784
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60_000, RESHAPE).astype('float32')
X_test = X_test.reshape(10_000, RESHAPE).astype('float32')

X_train /= 255.0
X_test /= 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPE,),
                             name='dense_layer_1', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPE,),
                             name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPE,),
                             name='dense_layer_3', activation='softmax'))
# Summary of the model
print(model.summary())
# Compiling the model
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Training the model
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
# Test Accuracy: 0.9793999791145325
print('Test Accuracy:', test_acc)

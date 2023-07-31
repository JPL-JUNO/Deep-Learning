"""
@Description: A real example: recognizing handwritten digits
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-31 09:26:37
"""
from tensorflow import keras
import tensorflow as tf
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 0
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = .2  # how much train is reserved for validation
RESHAPE = 784
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60_000, RESHAPE).astype('float32')
X_test = X_test.reshape(10_000, RESHAPE).astype('float32')

X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPE,),
                             name='dense_layer', activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test Accuracy:', test_acc)
print(model)

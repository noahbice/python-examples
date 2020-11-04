import numpy as np
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #suppresses tf outputs
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt

verbose = True

###load and visualize MNIST hand-written digit dataset
(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
num_train = x_train.shape[0]
num_test = x_test.shape[0]
print('{} training examples'.format(num_train))
print('{} testing examples'.format(num_test)) 

if verbose:
    print('---After loading---')
    print('Shapes: ', x_train.shape, y_train.shape)
    print('Min/max: ', np.min(x_train[0:10]), np.max(x_train[0:10]))
    n = 7
    fig, axs = plt.subplots(1,n)
    for i in range(n):
        axs[i].imshow(x_train[i], cmap='gray')
        axs[i].set_title(str(y_train[i]))
        axs[i].set_xticks([]); axs[i].set_yticks([])
    plt.show()

#quit()

###flatten, scale, change to one-hot encoding
x_train = x_train.reshape(num_train, -1); x_test =  x_test.reshape(num_test, -1)

x_train = x_train.astype('float32'); x_test = x_test.astype('float32')
x_train = x_train/255; x_test = x_test/255

y_train = tf.one_hot(y_train, 10); y_test = tf.one_hot(y_test, 10)

if verbose:
    print('---After preprocessing---')
    print('Shapes: ', x_train.shape, y_train.shape)
    print('Min/max: ', np.min(x_train[0:10]), np.max(x_train[0:10]))

#quit()

###define model, objective function, optimizer
feed_forward = k.Sequential()
feed_forward.add(k.layers.Dense(32, input_shape=(784,), activation='relu'))
feed_forward.add(k.layers.Dropout(0.5))
feed_forward.add(k.layers.Dense(10, activation='sigmoid'))
    
    
feed_forward.compile(loss=k.losses.categorical_crossentropy,
    optimizer=k.optimizers.Adam(),
    metrics=['categorical_accuracy'])

if verbose:
    feed_forward.summary()

#quit()

###train model
history = k.callbacks.History()
feed_forward.fit(x_train, y_train, batch_size=256, epochs=10, \
    verbose=verbose, validation_data=(x_test, y_test), callbacks=[history])

###evaluate performance
results = feed_forward.evaluate(x_test, y_test)
h = history.history
print('---Testing Results---')
print('Accuracy: ', results[1])
fig, axs = plt.subplots(1)
plt.plot(np.array(h['loss'])/6, label='training loss')
plt.plot(h['val_loss'], label='validation loss')
plt.ylabel('CCE Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

h = 5; w = 5
predictions = feed_forward.predict(x_test[0:h*w])
fig, axs = plt.subplots(h,w)
for i in range(h):
    for j in range(w):
        axs[i,j].imshow(x_test[i+h*j].reshape(28,28), cmap='gray')
        axs[i,j].set_title('Pred: ' + str(np.argmax(predictions[i+h*j])) \
            + '; True: ' + str(np.argmax(y_test[i+h*j])))
        axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
plt.tight_layout()
plt.show()







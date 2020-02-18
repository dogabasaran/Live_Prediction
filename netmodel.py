import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

'''
Builds a simple convolutional neural network
Trains the network after 
'''

def buildModel(print_summary=False):
    '''
    3 2-Dimensional Convolutional layers, 2 max-pools.
    '''
    model = keras.Sequential([
                    keras.layers.Conv2D(32, (3, 3), activation='relu',
                                        input_shape=(30, 30, 1)),
                    keras.layers.MaxPooling2D((2, 2)),
                    keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    keras.layers.MaxPooling2D((2, 2)),
                    keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(3),
                    keras.layers.Softmax()])
    if print_summary:
        # Prints out the model summary
        model.summary()
    return model


def train(model):
    # Percentage of data to be the validation set
    VAL_PERCENTAGE = 0.2
    # Number of epochs to run
    EPOCHS = 10
    BATCH_SIZE = 2
    # Path where model weights will be saved
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #Defines model optimizer and loss
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Loads training data
    data = np.load('training_data.npy', allow_pickle=True)
    # Converts data into shape suitable for tensorflow model, converts them to tensors
    X = tf.convert_to_tensor([np.expand_dims(item[0], -1) for item in data])
    y = tf.convert_to_tensor([item[1] for item in data])
    # Split data into training and validation sets
    val_size = int(VAL_PERCENTAGE*len(X))
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    test_X = X[-val_size:]
    test_y = y[-val_size:]
    
    # Model Checkpoint for saving weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    # Trains network
    history = model.fit(train_X, train_y, 
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(test_X, test_y),
                        callbacks=[cp_callback])

    # Evaluated model with validation set
    evaluateModel(model, history, test_X, test_y)
    print("Model weights saved.")
    


def evaluateModel(model, history, test_X, test_y):
    # Prints out graph of accuracy metric over time
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)

def predict(model, img):
    '''
    Used if you want to predict single images. Not for live prediction.
    '''
    ckpt_path = "training_1/cp.ckpt"
    # Loads saved weights onto model
    model.load_weights(ckpt_path)
    # Preprocesses input image: resize, normalize, reshapes and makes tensor
    input_img = cv2.resize(img, (30,30), interpolation=cv2.INTER_AREA)
    input_img = img / 255.0
    X = tf.convert_to_tensor([np.expand_dims(input_img,-1)])
    prediction = model.predict(X)
    # Shows test image and print's out prediction on terminal.
    cv2.imshow('test image', img)
    cv2.waitKey(0)
    print(np.argmax(prediction))

if __name__ == "__main__":
    model = buildModel(print_summary=True)
    train(model)

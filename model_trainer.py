#!/usr/bin/env python

"""model_trainer.py: Makes models that predict clothes based on images"""

__author__ = "Aurélien Bück-Kaeffer"
__license__ = "MIT"
__version__= "0.0.1"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras import layers
from keras.models import Sequential, clone_model
from keras.layers.core import Dense, Dropout, Activation, Flatten

def show_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

def create_model():
	"""Creates the model"""
	model = Sequential(
		[
			layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", input_shape = (28,28, 1), activation='relu'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.MaxPooling2D((2, 2)),
			layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
			layers.Flatten(),
			layers.Dense(64, activation = "linear"),
			layers.Dense(10, activation = "softmax")
		]
	)
	model.compile(loss='categorical_crossentropy', optimizer='Adam',  metrics = ['accuracy'])
	return model

def load_labels(filename):
	"""Loads and one hot encodes the labels"""
	df = pd.read_csv(filename)
	raw_labels = df["label"].values
	labels = np.zeros((raw_labels.size, raw_labels.max()+1))
	labels[np.arange(raw_labels.size), raw_labels] = 1
	return labels

train_images = np.load("train_images.npy")#Load training data
#show_image(train_images[x]) # 0 is the index of the training image you want to display

number_of_models = 25
models = []
for x in range(number_of_models):#Create n models
	models.append(create_model())

train_labels = load_labels("train_labels.csv")

for x, model in enumerate(models):#Train all the models
	val_split = 1/number_of_models
	print(f"Model {x+1}. Val from {int(x*val_split*len(train_images))} to {int((x+1)*val_split*len(train_images))}")
	val_set = train_images[int(x*val_split*len(train_images)):int((x+1)*val_split*len(train_images))]
	val_labels = train_labels[int(x*val_split*len(train_images)):int((x+1)*val_split*len(train_images))]
	train_set = np.concatenate(( train_images[:int(x*val_split*len(train_images))], train_images[int((x+1)*val_split*len(train_images)):]))
	train_lb = np.concatenate(( train_labels[:int(x*val_split*len(train_images))], train_labels[int((x+1)*val_split*len(train_images)):]))
	history = model.fit(train_set, train_lb, validation_data=(val_set, val_labels), epochs = 5)

	"""plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()"""

def make_prediction(models, data):
	"""Takes the prediction of every model and gives a final answer based on majority vote"""
	predictions = []
	#take the prediction from every model
	for x in models:
		predictions.append(x.predict(x_test).T)
	predictions = np.array(predictions).T
	predictions = np.transpose(predictions, axes=(0, 2, 1))
	predictions = np.argmax(predictions, axis=2)

	#Pick the final prediction based on majority vote
	final_result = []
	for x in predictions:
		final_result.append(np.bincount(x).argmax())
	final_result = np.array(final_result)

	return final_result

x_test = np.load('test_images.npy')
y_test = make_prediction(models, x_test)
print(y_test)

df_test = pd.read_csv('sample_submission.csv')
df_test['label'] = y_test
df_test.to_csv('submission.csv', index=False)
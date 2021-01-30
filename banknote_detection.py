#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt

data = loadtxt('data_banknote_authentication.txt', delimiter=',')

inputs = data[:,0:4]
outputs = data[:,4]

model = Sequential()

model.add(Dense(6, input_dim=4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=100, batch_size=10)

_, accuracy = model.evaluate(inputs, outputs)
print(accuracy)

#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
import numpy as np

data = loadtxt('haberman.data', delimiter=',')

for i in range(306):
    if(data[i,3]==1.0):
        data[i,3]=0
    else:
        data[i,3]=1



inputs = data[:,0:3]
outputs = data[:,3]

model = Sequential()

model.add(Dense(6, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(inputs, outputs, epochs=30, batch_size=10)

_, accuracy = model.evaluate(inputs, outputs)
print(accuracy)
#print(inputs[6:8])
#print(model.predict(np.array(inputs[6:8])))
#print(model.get_weights())
#weights = model.get_weights()
#model.set_weights(weights)
#print(model.get_weights())

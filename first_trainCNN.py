import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.backend import clear_session
from keras.layers  import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy

(train_X , train_y), (test_X , test_y) = mnist.load_data("mymnist.data")

test_X = test_X.reshape(test_X.shape[0],28,28,1)
train_X = train_X.reshape(train_X.shape[0] ,  28,28,1)
test_X = test_X.astype("float32")
train_X = train_X.astype("float32")


test_y = to_categorical(test_y)
train_y = to_categorical(train_y)

model = Sequential()
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units = 512  , activation = 'relu'))
model.add(Dense(units=256  , activation = 'relu'))
model.add(Dense(units=128 , activation = 'relu'))
model.add(Dense(units=10  , activation = 'softmax'))

model.compile( optimizer= "RMSprop" , loss='categorical_crossentropy', 
             metrics=['accuracy'] )
fit_model = model.fit(train_X ,  train_y , epochs = 5 , verbose =  False)


text = fit_model.history
accuracy = text['accuracy'][1] * 100
accuracy = int(accuracy)
f= open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy for the model is : " , accuracy ,"%")
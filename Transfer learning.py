from keras.applications import resnet50
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

model=resnet50.ResNet50(weights='imagenet' ,input_shape=(224,224,3) , include_top=False)

for layers in model.layers:
    layers.trainable = False


t_layer=model.output
t_layer=Flatten()(t_layer)
t_layer=Dense(units=1024,activation='relu')(t_layer)
t_layer=Dense(units=1024,activation='relu')(t_layer)
t_layer=Dense(units=512,activation='relu')(t_layer)
t_layer=Dense(units=3,activation='softmax')(t_layer)

n_model=Model(inputs= model.input , outputs=t_layer)


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/train_set/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

n_model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("face_detect.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [earlystop, checkpoint]

nb_train_samples = 300
nb_validation_samples = 108


epochs = 3
batch_size = 16

history = n_model.fit_generator(
    training_set,
    steps_per_epoch = nb_train_samples // batch_size ,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = test_set,
    validation_steps = nb_validation_samples // batch_size )
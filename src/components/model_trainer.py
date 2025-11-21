import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.logger import logging
from src.exception import CustomException


class ModelTrainer():

    def __init__(self,learning_rate=0.001):
        self.learning_rate=learning_rate
        os.makedirs("artifacts",exist_ok=True) # Stocke tous les résultats générés automatiquement par le pipeline

    def build_model(self):
        try:
            logging.info("Building CNN model")

            model = Sequential()

            model.add(Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)))
            model.add(MaxPooling2D())

            model.add(Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)))
            model.add(MaxPooling2D(2,2))

            model.add(Conv2D(128, (3,3), activation='relu'))
            model.add(MaxPooling2D(2,2))

            model.add(Flatten())

            model.add(Dense(128))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.3))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer=Adam(learning_rate=self.learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
            
            logging.info("Model compiled successfully")
            return model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def train(self,train_generator,val_generator):
        try:
            logging.info("Starting model training")

            model = self.build_model()

            checkpoint_path = "artifacts/best_model.keras"

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            checkpoint = ModelCheckpoint(filepath="artifacts/best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

            history = model.fit(train_generator,epochs=30,validation_data=val_generator,callbacks=[early_stopping,checkpoint])

            logging.info("Model training completed successfully")

            return model, checkpoint_path, history

        except Exception as e:
            raise CustomException(e,sys)      
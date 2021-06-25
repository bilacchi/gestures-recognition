#%% Import Packages
import json
import zipfile
import argparse
import pandas as pd
import tensorflow as tf

from models import *
from generator import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%% Parser
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser(description='Tensorflow Jester Training using JPEG')
parser.add_argument('--config', '-c', help='json config file path', default='./config.json')
parser.add_argument('--eval_only', '-e', default=False, type=str2bool, help="evaluate trained model on validation data.")
parser.add_argument('--resume', '-r', default=False, type=str2bool, help="resume training from given checkpoint.")
parser.add_argument('--zip', '-z', default=True, type=str2bool, help="use zip file to train the model.")
args = parser.parse_args()

#%% Load Main Configs
with open(args.config) as jfile:
    config = json.load(jfile)

#%% Train 
def main():
    global args
    model = build_model(shape=(config['nb_frames'], 100, 100, 3), n_classes=config['num_classes'], convnet=eval(config['convnet']))
    
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    if args.resume:
        try: model.load_weights(config['model_checkpoint'])
        except: raise "It was not possible to resume your model"

    datagen = ImageDataGenerator(rescale=1./255.,
                            rotation_range=15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.1,
                            zoom_range=0.2,
                            horizontal_flip=True) # Image Augmentation

    validate = pd.read_csv(config['validation_dataset'], sep=';', header=None)
    if args.zip:
        ZIPFILE = zipfile.ZipFile(config['zip_file'])
        validationGenerator = VideoFrameGeneratorZip(zipf=ZIPFILE,
                            files=validate[0],
                            labels=validate[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])
    else:
        validationGenerator = VideoFrameGenerator(path=config['images_folder'],
                            files=validate[0],
                            labels=validate[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])

    if args.eval_only:
        validate(model, validationGenerator)
        return

    train = pd.read_csv(config['train_dataset'], sep=';', header=None)
    test = pd.read_csv(config['test_dataset'], sep=';', header=None)

    #%% Create Data generators
    if args.zip:
        trainGenerator = VideoFrameGeneratorZip(zipf=ZIPFILE,
                            files=train[0],
                            labels=train[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])

        testGenerator = VideoFrameGeneratorZip(zipf=ZIPFILE,
                            files=test[0],
                            labels=test[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])
    else:
        trainGenerator = VideoFrameGenerator(path=config['images_folder'],
                            files=train[0],
                            labels=train[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])

        testGenerator = VideoFrameGenerator(path=config['images_folder'],
                            files=test[0],
                            labels=test[1],
                            target_shape=(100,100),
                            transformation = datagen,
                            nb_frames=config['nb_frames'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f'{config["checkpoint"]}{config["model_name"]}.hdf5', verbose=1),
        tf.keras.callbacks.CSVLogger(f'{config["checkpoint"]}{config["model_name"]}-Log.csv', separator=',', append=True)
    ]

    history = model.fit(trainGenerator, validation_data = testGenerator, verbose = 1,
                        epochs = config['num_epochs'], callbacks = callbacks)

    return history

def validate(model, validation_data):
    pass
    
if __name__ == '__main__':
    main()
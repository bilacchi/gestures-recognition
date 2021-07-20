#%% Import Packages
import json
import argparse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from models import *
from generator import *
from keras_buoy.models import ResumableModel
from sklearn.metrics import confusion_matrix, cohen_kappa_score
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
    
    if args.resume:
        try: model = tf.keras.models.load_model(config['checkpoint']+config['model_name']+'.h5')
        except: raise "It was not possible to resume your model"
    
    else:
        model = build_model(shape=(config['nb_frames'], 100, 100, 3), n_classes=config['num_classes'], convnet=eval(config['convnet']))
        model.compile(optimizer='Adam',
                    loss='CategoricalCrossentropy',
                    metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top@1'),
                            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5')])
    
    if args.eval_only:
        return validate(model)
        
    datagen = ImageDataGenerator(rescale=1./255.,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                rotation_range=5) # Image Augmentation

    train = pd.read_csv(config['train_dataset'], sep=';', header=None)
    test = pd.read_csv(config['test_dataset'], sep=';', header=None)

    trainGenerator = VideoFrameGenerator(path=config['data_path'],
                    files=train[0],
                    labels=train[1],
                    target_shape=(100,100),
                    transformation = datagen,
                    nb_frames=config['nb_frames'],
                    batch_size=config['batch_size'])

    testGenerator = VideoFrameGenerator(path=config['data_path'],
                    files=test[0],
                    labels=test[1],
                    target_shape=(100,100),
                    transformation = datagen,
                    nb_frames=config['nb_frames'],
                    batch_size=config['batch_size'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=.2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(config['checkpoint']+config['model_name']+'-{epoch:02d}.h5')
    ]

    resumable_model = ResumableModel(model, save_every_epochs=1, to_path=f'{config["checkpoint"]}{config["model_name"]}.h5')
    history = resumable_model.fit(trainGenerator,  validation_data = testGenerator, verbose=1, epochs=config['num_epochs'], callbacks=callbacks) # >= 12.5% at least
    plotHistory(history)
    
def plotHistory(history):
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].plot(history['loss'], color='firebrick')
    axs[0].plot(history['val_loss'], color='teal')
    axs[0].grid(color='grey', linestyle='dashed', alpha=.3)
    axs[0].set_ylabel('Loss')

    axs[1].plot(history['top@1'], color='firebrick', label='train')
    axs[1].plot(history['val_top@1'], color='teal', label='test')
    axs[1].grid(color='grey', linestyle='dashed', alpha=.3)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Top@1')
    axs[1].legend(loc='best')

    axs[2].plot(history['top@5'], color='firebrick', label='train')
    axs[2].plot(history['val_top@5'], color='teal', label='test')
    axs[2].grid(color='grey', linestyle='dashed', alpha=.3)
    axs[2].set_ylabel('Top@5')
    axs[2].legend(loc='best')

    plt.subplots_adjust(wspace=.27)
    plt.savefig(config['checkpoint']+config['model_name']+'.png', bbox_inches='tight', dpi=300)

def validate(model):
    validate = pd.read_csv(config['validation_dataset'], sep=';', header=None)
    validateGenerator = VideoFrameGenerator(path=config['data_path'],
                        files=validate[0],
                        labels=validate[1],
                        target_shape=(100,100),
                        transformation = ImageDataGenerator(rescale=1./255.),
                        nb_frames=config['nb_frames'],
                        batch_size=10,
                        shuffle=False)
    model.evaluate(validateGenerator)
    
    ypred = np.argmax(model.predict(validateGenerator), axis=1)
    ytrue = np.unique(validate[1], return_inverse=True)[1]

    print('Confusion Matrix\n', confusion_matrix(ytrue, ypred))
    print('Cohen Kappa\n', cohen_kappa_score(ytrue, ypred))
    
if __name__ == '__main__':
    main()
import io
import os
import zipfile
import numpy as np

from PIL import Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

class VideoFrameGenerator(Sequence):
    def __init__(
            self,
            files: list = [],
            labels: list = [],
            path: str = None,
            nb_frames: int = 5,
            batch_size: int = 16,
            target_shape: tuple = (224, 224),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            nb_channel: int = 3,
            *args,
            **kwargs):

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3), "should be only RGB or Grayscale"

        # shape size should be 2
        assert len(target_shape) == 2

        self.batch_size = batch_size
        self.nbframe = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = 'grayscale' if nb_channel == 1 else 'rgb'
        self.transformation = transformation

        self._random_trans = []
        self.files = files
        self.labels = labels
        self.path = path
        self.classes = np.unique(labels).tolist()

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)

        # to initialize transformations and shuffle indices
        if 'no_epoch_at_init' not in kwargs:
            self.on_epoch_end()

        self._current = 0
        self._framecounters = {}

    def next(self):
        """ Return next element"""
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def on_epoch_end(self):
        """ Called by Keras after each epoch """

        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            video = self.files[i]
            classname = self.labels[i]

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            
            frames = self._get_frames(video, nbframe, shape)
            if frames is None:
                continue # avoid failure, nevermind that video...

            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_frames(self, video, nbframe, shape):
        video_path = os.path.join(self.path, str(video))
        cap = os.listdir(video_path)
        total_frames = len(cap)

        jitter = np.random.randint(0, total_frames-nbframe+1) if total_frames > nbframe else 0
        frames = []

        for iframe in cap[jitter:nbframe+jitter]:
            image_path = os.path.join(video_path, str(iframe))
            loaded = load_img(image_path, color_mode=self.nb_channel, target_size=shape)
            frame = img_to_array(loaded)

            # keep frame
            frames.append(frame)
            if len(frames) == nbframe:
                break
        
        while True:
            if len(frames) < nbframe:
                frames.append(frame)
            elif len(frames) > nbframe:
                frames.pop(-1)
            else:
                break
        return np.array(frames)


class VideoFrameGeneratorZip(VideoFrameGenerator):
    def __init__(self,
                zipf: zipfile.ZipFile = None,
                **kwargs):
        super().__init__(**kwargs)

        assert isinstance(zipf, zipfile.ZipFile), "You must provide a ZipFile"

        self.zipfile = zipf
        self.path = self.zipfile.namelist()[0]
        self.nb_channel = 'RGB' if self.nb_channel == 'rgb' else 'L'

    def _get_frames(self, video, nbframe, shape):
        video = str(video)
        video_path = os.path.join(self.path, video)
        cap = [frame for frame in self.zipfile.namelist() if frame if frame.startswith(video_path + os.path.sep)\
            and frame.endswith(('.jpg', '.JPG',  '.jpeg', '.JPEG'))]
        total_frames = len(cap)
        
        jitter = np.random.randint(0, total_frames-nbframe+1) if total_frames > nbframe else 0
        frames = []

        for iframe in cap[jitter:nbframe+jitter]:
            frame = Image.open(io.BytesIO(self.zipfile.read(iframe)))
            frame = frame.convert(self.nb_channel)
            frame = frame.resize(shape, Image.NEAREST)
            frame = np.array(frame)

            # keep frame
            frames.append(frame)
            if len(frames) == nbframe:
                break
        
        while True:
            if len(frames) < nbframe:
                frames.append(frame)
            elif len(frames) > nbframe:
                frames.pop(-1)
            else:
                break

        return np.array(frames)
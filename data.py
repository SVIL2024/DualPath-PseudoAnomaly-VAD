import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
import random

# from PIL import Image

# print(cv2.__version__)

rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, grayscale=False):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if grayscale:
        image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(filename)
    if image_decoded is None:
        raise RuntimeError(f"Failed to decode image: {filename}")
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    image_mean = float(image_resized.mean())
    image_std = max(float(image_resized.std()), 1e-6)
    image_resized = (image_resized - image_mean) / image_std
    return image_resized



class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=16,
                 img_extension='.jpg', dataset='ped2', jump=[2], hold=[2],
                 return_normal_seq=False, train=True, frame_cache_size=0):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames
        self.jump = jump
        self.extension = img_extension
        self.dataset = dataset
        self.hold = hold
        self.frame_cache_size = max(0, frame_cache_size)
        self.frame_cache = OrderedDict()
        self.setup()
        self.samples, self.background_models = self.get_all_samples()

        self.return_normal_seq = return_normal_seq  # for fast and slow moving

        if self.dataset == 'ped2':
            self.gray = True
        else :
            self.gray = False

    def load_frame(self, filename):
        if self.frame_cache_size == 0:
            return np_load_frame(filename, self._resize_height, self._resize_width,
                                 grayscale=self.gray)

        image = self.frame_cache.pop(filename, None)
        if image is None:
            image = np_load_frame(filename, self._resize_height, self._resize_width,
                                  grayscale=self.gray)
        self.frame_cache[filename] = image

        if len(self.frame_cache) > self.frame_cache_size:
            self.frame_cache.popitem(last=False)
        return image


    def setup(self):
        videos = [
            video for video in glob.glob(os.path.join(self.dir, '*/'))
            if not os.path.basename(os.path.normpath(video)).lower().endswith('_gt')
        ]
        for video in sorted(videos):
            video_name = os.path.basename(os.path.normpath(video))
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.extension))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        background_models = []
        videos = [
            video for video in glob.glob(os.path.join(self.dir, '*/'))
            if not os.path.basename(os.path.normpath(video)).lower().endswith('_gt')
        ]
        for video in sorted(videos):
            video_name = os.path.basename(os.path.normpath(video))

            for i in range(len(self.videos[video_name]['frame']) - self._num_frames + 1):
                frames.append(self.videos[video_name]['frame'][i])

        return frames, background_models


    def __getitem__(self, index):
        # index = 8
        video_name = os.path.basename(os.path.dirname(self.samples[index]))
        # if self.dataset == 'shanghai' and 'training' in self.samples[index]:
        if self.dataset == 'shanghai':
            frame_name = int(os.path.splitext(os.path.basename(self.samples[index]))[0]) - 1
        else:
            # frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])
            # frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])
            frame_name = int(os.path.splitext(os.path.basename(self.samples[index]))[0]) - 1

        batch = []
        for i in range(self._num_frames):
            image = self.load_frame(self.videos[video_name]['frame'][frame_name + i])

            if self.transform is not None:
                batch.append(self.transform(image))

        return np.stack(batch, axis=1)

    def __len__(self):
        return len(self.samples)


class Reconstruction3DDataLoaderJump(Reconstruction3DDataLoader):
    def __getitem__(self, index):
        # index = 8
        video_name = os.path.basename(os.path.dirname(self.samples[index]))
        if self.dataset == 'shanghai' and 'training' in self.samples[index]:  # bcos my shanghai's start from 1
            frame_name = int(os.path.splitext(os.path.basename(self.samples[index]))[0]) - 1
        else:
            frame_name = int(os.path.splitext(os.path.basename(self.samples[index]))[0]) - 1

        batch = []
        normal_batch = []
        jump = random.choice(self.jump)

        retry = 0
        while len(self.videos[video_name]['frame']) <= frame_name + (self._num_frames-1) * jump and retry < 10:
            # reselect the frame_name
            max_start = max(1, len(self.videos[video_name]['frame']) - (self._num_frames - 1) * jump)
            frame_name = np.random.randint(max_start)
            retry += 1

        for i in range(self._num_frames):
            image = self.load_frame(
                self.videos[video_name]['frame'][
                    min(frame_name + i * jump, len(self.videos[video_name]['frame']) - 1)
                ]
            )

            if self.transform is not None:
                batch.append(self.transform(image))

        if self.return_normal_seq:
            for i in range(self._num_frames):
                image = self.load_frame(
                    self.videos[video_name]['frame'][
                        min(frame_name + i, len(self.videos[video_name]['frame']) - 1)
                    ]
                )

                if self.transform is not None:
                    normal_batch.append(self.transform(image))
            return np.stack(batch, axis=1), np.stack(normal_batch, axis=1)

        else:
            return np.stack(batch, axis=1)


import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from src.model.stageI.stageI_gan import StageIGAN
from src.model.stageII.stageII_gan import StageIIGAN
from src.utils.data_utils import normalize_images

percentage_train = 0.8

caption_embeds = np.load('data/flowers/caption_embeds.npy')
low_res = []
high_res = []


files = glob.glob('data/flowers/images/*')
train_num = int(len(files)*percentage_train)

indices = np.arange(len(files))
np.random.shuffle(indices)

files = [files[idx] for idx in indices]
caption_embeds = caption_embeds[indices]

image_files = files[:train_num]

train_embeds = caption_embeds[:train_num]
test_embeds = caption_embeds[train_num:]

run_dir = 'runs/'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

start_epoch = 34

stageI_gan = StageIGAN(run_dir, generator_lr=0.001)
stageI_gan.load()
stageI_gan.train(
    image_files,
    train_embeds,
    test_embeds,
    500,
    start_epoch=start_epoch,
    batch_size=32
)
stageII_gan = StageIIGAN(run_dir, generator_lr=0.001)

start_epoch = 0

stageII_gan.train(
    image_files,
    train_embeds,
    test_embeds,
    500,
    start_epoch=start_epoch,
    batch_size=4
)
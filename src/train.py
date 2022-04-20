import numpy as np
import os
import glob
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from src.model.stageI.stageI_gan import StageIGAN
from src.model.stageII.stageII_gan import StageIIGAN
from src.utils.data_utils import normalize_images

percentage_train = 0.7

caption_embeds = np.load('data/flowers/caption_embeds.npy')
low_res = []
high_res = []

files = glob.glob('data/flowers/images/*')
train_num = int(len(files)*percentage_train)


image_files = files[:train_num]

train_embeds = caption_embeds[:train_num]
test_embeds = caption_embeds[train_num:]

run_dir = 'runs/'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stageI_gan = StageIGAN(run_dir)

stageI_gan.train(
    image_files,
    train_embeds,
    test_embeds,
    50,
    32
)
stageII_gan = StageIIGAN(run_dir)

stageII_gan.train(
    image_files,
    train_embeds,
    test_embeds,
    50,
    4
)
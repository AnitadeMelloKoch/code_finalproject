import numpy as np
import os
import glob
from src.model.stageII.stageII_gan import StageIIGAN
import matplotlib.pyplot as plt
from src.utils.training_utils import adjust_image
from textwrap import wrap

percentage_test = 0.005
save_dir = 'test/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

caption_embeds = np.load('data/flowers/caption_embeds.npy')

files = glob.glob('data/flowers/captions/*')

indices = np.arange(len(files))
np.random.shuffle(indices)

files = [files[idx] for idx in indices]
caption_embeds = caption_embeds[indices]

test_num = int(len(files)*percentage_test)

test_files = files[:test_num]

gan = StageIIGAN('runs/ccv_data/runs/')
gan.load()


for i, file_name in enumerate(test_files):
    file = open(file_name, 'r')
    lines = file.readlines()
    for j, line in enumerate(lines):
        embedding = caption_embeds[i, j]
        
        low_res, high_res = gan.evaluate(embedding)

        low_res = adjust_image(low_res)
        high_res = adjust_image(high_res)
        
        image, (ax_small, ax_big) = plt.subplots(1, 2)

        ax_small.imshow(low_res)
        ax_small.axis("off")

        ax_big.imshow(high_res)
        ax_big.axis("off")

        image.suptitle("\n".join(wrap(line, 60)), wrap=True)

        plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(i, j)), bbox_inches='tight')

        plt.close('all')


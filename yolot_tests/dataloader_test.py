'''
    This is a script for testing the BMOTS Dataloader.
'''

import torch
import torchvision as tv
from yolot.BMOTSDataset import BMOTSDataset, collate_fn
from ultralytics.data.build import InfiniteDataLoader
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

default_dataset_path = "C:\\Users\\dylan\\Documents\\Data\\BDD100k_MOT202\\bdd100k"

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', type=bool, default=True, help='Show training results')
    parser.add_argument('--data', type=str, default=default_dataset_path, help='Number of Epochs to be trained')
    return parser

def main_func(args):
    # Read in args
    vis = args.vis
    data_path = args.data

    # Initialize Dataset
    dataset = BMOTSDataset(data_path,"train")

    # Intialize Dataloader
    loader = InfiniteDataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, drop_last=False, pin_memory=False)

    # Load and visualize data
    for i,batch in enumerate(loader):
        print(batch[0]['im_file'][0])
        show(batch[0]['img'][1,:,:,:])
        if i > 3:
            break
    plt.show()
    print("done!")




def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

if __name__ == '__main__':
    args = init_parser().parse_args()
    main_func(args)


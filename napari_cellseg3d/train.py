from pathlib import Path
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.config import WNetTrainingWorkerConfig, WandBConfig, WeightsInfo, PRETRAINED_WEIGHTS_DIR
import torch
import numpy as np
from PIL import Image
import os
import shutil
from napari_cellseg3d import config, utils
from monai.transforms import LoadImaged, Compose
from monai.data import DataLoader, Dataset
import cv2
import logging

utils.LOGGER.setLevel(logging.DEBUG)

training_source_2 = "./gdrive/MyDrive/ComputerScience/WesternResearch/data/slice_origin"
training_source = "/content/slice"
if os.path.exists(training_source):
    shutil.rmtree(training_source)
os.mkdir(training_source)


def preprocess_to_3d_tiles(infile, out_prefix, ch, tile_width, clamp_max, clamp_min=0):
    """Preprocess a 4d slice (ch, z, y, x) file by selecting one of its channel and turn the volume into
    3d tiles of a given size;
    If the tile_width is None, then the slice is not tiled"""
    im = np.load(infile)[ch]
    clamp_range = clamp_max - clamp_min
    im = np.clip((im - clamp_min) / clamp_range, 0, 1)
    if tile_width is None:
        np.save(f'{out_prefix}.npy', im)
    else:
        for i in range(im.shape[0] // tile_width):
            istart = i * tile_width
            for j in range(im.shape[1] // tile_width):
                jstart = j * tile_width
                for k in range(im.shape[2] // tile_width):
                    kstart = k * tile_width
                    sli = im[istart:istart + tile_width, jstart:jstart + tile_width, kstart:kstart + tile_width]
                    np.save(f'{out_prefix}_{i}_{j}_{k}.npy', sli)


model_path = "./gdrive/MyDrive/ComputerScience/WesternResearch/data/WNET_TRAINING_RESULTS"
do_validation = False
use_default_advanced_parameters = False

batch_size = 4
learning_rate = 2e-5
num_classes = 10
weight_decay = 0.01
validation_frequency = 2
intensity_sigma = 1.0
spatial_sigma = 4.0
ncuts_radius = 2

src_pth = training_source
train_data_folder = Path(src_pth)
results_path = Path(model_path)
results_path.mkdir(exist_ok=True)
eval_image_folder = Path(src_pth)
eval_label_folder = Path(src_pth)

eval_dict = c.create_eval_dataset_dict(
    eval_image_folder,
    eval_label_folder,
) if do_validation else None


def create_dataset(folder):
    images_filepaths = utils.get_all_matching_files(folder, pattern={'.npy', })
    # images_filepaths = images_filepaths.get_unsupervised_image_filepaths()

    data_dict = [{"image": str(image_name)} for image_name in images_filepaths]
    return data_dict


WANDB_INSTALLED = False

train_config = None
wandb_config = None

def init_configs(args):
    number_of_epochs = args['number_of_epochs']
    n_cuts_weight = args['n_cuts_weight']
    rec_loss_weight = args['rec_loss_weight']
    rec_loss = args['rec_loss']
    global train_config, wandb_config
    train_config = WNetTrainingWorkerConfig(
        device="cuda:0",
        max_epochs=number_of_epochs,
        learning_rate=2e-5,
        validation_interval=2,
        batch_size=batch_size,
        num_workers=2,
        weights_info=WeightsInfo(),
        results_path_folder=str(results_path),
        train_data_dict=create_dataset(train_data_folder),
        eval_volume_dict=eval_dict,
    ) if use_default_advanced_parameters else WNetTrainingWorkerConfig(
        device="cuda:0",
        max_epochs=number_of_epochs,
        learning_rate=learning_rate,
        validation_interval=validation_frequency,
        batch_size=batch_size,
        num_workers=2,
        weights_info=WeightsInfo(),
        results_path_folder=str(results_path),
        train_data_dict=create_dataset(train_data_folder),
        eval_volume_dict=eval_dict,
        # advanced
        num_classes=num_classes,
        weight_decay=weight_decay,
        intensity_sigma=intensity_sigma,
        spatial_sigma=spatial_sigma,
        radius=ncuts_radius,
        reconstruction_loss=rec_loss,
        n_cuts_weight=n_cuts_weight,
        rec_loss_weight=rec_loss_weight,
    )
    wandb_config = WandBConfig(
        mode="disabled" if not WANDB_INSTALLED else "online",
        save_model_artifact=False,
    )


def train_model():
    worker = c.get_colab_worker(worker_config=train_config, wandb_config=wandb_config)
    for epoch_loss in worker.train():
        continue

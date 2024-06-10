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
from napari_cellseg3d.dev_scripts.colab_training import print_mem_usage
import logging
utils.LOGGER.setLevel(logging.DEBUG)

training_source_2 = "./gdrive/MyDrive/ComputerScience/WesternResearch/data/slice_origin"
training_source = "/content/slice"
if os.path.exists(training_source):
    shutil.rmtree(training_source)
os.mkdir(training_source)
im = np.load(f'{training_source_2}/slice.npy')[0]
im = im / im.max()
BLOCK_WIDTH = 64
for ax_idx in range(3):
    for i in range(im.shape[0] // BLOCK_WIDTH):
        istart = i * BLOCK_WIDTH
        for j in range(im.shape[1] // BLOCK_WIDTH):
            jstart = j * BLOCK_WIDTH
            for k in range(im.shape[2] // BLOCK_WIDTH):
                kstart = k * BLOCK_WIDTH
                sli = im[istart:istart + BLOCK_WIDTH, jstart:jstart + BLOCK_WIDTH, kstart:kstart + BLOCK_WIDTH]
                np.save(f'{training_source}/slice_{i}_{j}_{k}.npy', sli)
model_path = "./gdrive/MyDrive/ComputerScience/WesternResearch/data/WNET_TRAINING_RESULTS"
do_validation = False
number_of_epochs = 50
use_default_advanced_parameters = False

batch_size = 4
learning_rate = 2e-5
num_classes = 10
weight_decay = 0.01
validation_frequency = 2
intensity_sigma = 1.0
spatial_sigma = 4.0
ncuts_radius = 2
rec_loss = "MSE"
n_cuts_weight = 0.5
rec_loss_weight = 0.005

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

from napari_cellseg3d.code_models.models.wnet.model import WNet
import torch
import numpy as np
from napari_cellseg3d.utils import remap_image
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from napari_cellseg3d.func_variants import normalize_inplace


DEVICE = 'cuda'


def create_model(train_config, weights):
    model = WNet(
        in_channels=train_config.in_channels,
        out_channels=train_config.out_channels,
        num_classes=train_config.num_classes,
        dropout=train_config.dropout,
    )
    model.to(DEVICE)
    model.load_state_dict(weights, strict=True)
    return model


def inference_on(model, image_files):
    """
    :param model:
    :param image_files: one file or many files
    :return:
    """
    with torch.no_grad():
        model.eval()
        for _k, val_data_file in enumerate(image_files):
            val_data = np.float32(np.load(val_data_file)[None, None, :])
            val_inputs = torch.from_numpy(val_data).to(DEVICE)

            # normalize val_inputs across channels
            for i in range(val_inputs.shape[0]):
                for j in range(val_inputs.shape[1]):
                    im_max = val_inputs.max()
                    im_min = val_inputs.min()
                    normalize_inplace(val_inputs[i, j], im_max=im_max, im_min=im_min)
            print(f"Val inputs shape: {val_inputs.shape}")
            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=[64, 64, 64],
                sw_batch_size=1,
                predictor=model.forward_encoder,
                overlap=0.1,
                mode="gaussian",
                sigma_scale=0.01,
                progress=True,
            )
            # val_decoder_outputs = sliding_window_inference(
            #     val_outputs,
            #     roi_size=[64, 64, 64],
            #     sw_batch_size=1,
            #     predictor=model.forward_decoder,
            #     overlap=0.1,
            #     mode="gaussian",
            #     sigma_scale=0.01,
            #     progress=True,
            # )
            val_outputs = val_outputs
            yield val_outputs

import argparse
from PIL import Image
from lib import image_utils
from lib import metrics
from lib.diffc.encode import encode
from lib.diffc.denoise import denoise
from lib.diffc.rcc.gaussian_channel_simulator import GaussianChannelSimulator
from easydict import EasyDict as edict
import yaml
from pathlib import Path
from lib.blip import BlipCaptioner


import pandas as pd

###############################################################################
## Parse arguments
###############################################################################

parser = argparse.ArgumentParser(
    description="Evaluate the DiffC compression algorithm on an image or folder of images."
)
parser.add_argument(
    "--config",
    help="Path to the compression config file. For example, config/SD-1.5-no-prompt.yaml",
)
parser.add_argument(
    "--image_path", default=None, help="Path to a single image to compress"
)
parser.add_argument(
    "--image_dir",
    default=None,
    help="Path to a directory containing one or more images to compress",
)
parser.add_argument(
    "--output_dir", help="Directory to output the compression results to."
)

args = parser.parse_args()

with open(args.config, "r") as file:
    config = edict(yaml.safe_load(file))

###############################################################################
## Get image paths and optionally BLIP captions
###############################################################################

image_paths = []

if not bool(args.image_path) ^ bool(args.image_dir):
    raise ValueError("Must specify exactly one of --image_path or --image_dir")

if args.image_path:
    image_paths.append(Path(args.image_path))
else:
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    image_paths = list(map(Path, image_paths))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

captions = {}
if config.encoding_guidance_scale or config.denoising_guidance_scale:
    captioner = BlipCaptioner()
    captions = captioner.process_and_save(image_paths, output_dir)
    del captioner  # Free up GPU memory

###############################################################################
## Make GaussianChannelSimulator and LatentNoisePredictionModel
###############################################################################

gaussian_channel_simulator = GaussianChannelSimulator(
    config.max_chunk_size, config.chunk_padding
)

if config.model == "SD1.5":
    from lib.models.SD15 import SD15Model

    noise_prediction_model = SD15Model()
elif config.model == "SD2.1":
    from lib.models.SD21 import SD21Model

    noise_prediction_model = SD21Model()
elif config.model == "SDXL":
    from lib.models.SDXL import SDXLModel
    use_refiner = config.get("use_refiner", False)
    noise_prediction_model = SDXLModel(use_refiner=use_refiner)
elif config.model == 'Flux':
    from lib.models.Flux import FluxModel
    noise_prediction_model = FluxModel()
else:
    raise ValueError(f"Unrecognised model: {config.model}")

###############################################################################
## Evaluate on the provided images
###############################################################################

results_data = []

for image_path in image_paths:

    ## Load and preprocess the image
    ###########################################################################

    img_pil = Image.open(image_path)
    img_width, img_height = img_pil.size
    gt_pt = image_utils.pil_to_torch_img(img_pil)
    gt_latent = noise_prediction_model.image_to_latent(gt_pt)
    prompt = ""
    if config.encoding_guidance_scale or config.denoising_guidance_scale:
        prompt = captions[str(image_path)]

    noise_prediction_model.configure(
        prompt, config.encoding_guidance_scale, img_width, img_height
    )

    ## Encode the image
    ###########################################################################

    chunk_seeds_per_step, Dkl_per_step, noisy_recons, noisy_recon_step_indices = encode(
        gt_latent,
        config.encoding_timesteps,
        noise_prediction_model,
        gaussian_channel_simulator,
        config.manual_dkl_per_step,
        config.recon_timesteps,
    )

    ## Create reconstructions, save them to disk, evaluate metrics
    ###########################################################################

    noise_prediction_model.configure(
        prompt, config.denoising_guidance_scale, img_width, img_height
    )

    for noisy_recon, step_idx in zip(noisy_recons, noisy_recon_step_indices):

        bytes = gaussian_channel_simulator.compress_chunk_seeds(
            chunk_seeds_per_step[: step_idx + 1], Dkl_per_step[: step_idx + 1]
        )
        bpp = len(bytes) * 8 / (img_width * img_height)
        # TODO: add prompt length to bpp

        timestep = config.encoding_timesteps[step_idx]
        snr = noise_prediction_model.get_timestep_snr(timestep).item()

        recon_latent = denoise(
            noisy_recon, timestep, config.denoising_timesteps, noise_prediction_model
        )

        recon_img_pt = noise_prediction_model.latent_to_image(recon_latent)

        psnr = metrics.get_psnr(gt_pt, recon_img_pt)
        lpips = metrics.get_lpips(gt_pt, recon_img_pt)

        timestep_dir = output_dir / str(int(timestep)).zfill(3)
        timestep_dir.mkdir(exist_ok=True, parents=True)
        recon_path = timestep_dir / image_path.name
        image_utils.torch_to_pil_img(recon_img_pt).save(recon_path)

        results_data.append(
            {
                "gt_path": str(image_path),
                "recon_path": str(recon_path),
                "recon_step_idx": step_idx,
                "recon_timestep": timestep,
                "snr": snr,
                "bpp": bpp,
                "psnr": psnr,
                "lpips": lpips,
            }
        )

###############################################################################
## Write out metrics to a csv
###############################################################################

results_df = pd.DataFrame(data=results_data)
results_df.to_csv(output_dir / "results.csv")

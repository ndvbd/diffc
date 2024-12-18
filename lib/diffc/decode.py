import torch
from lib.diffc.p import P


@torch.no_grad()
def decode(
    latent_shape,
    latent_device,
    latent_dtype,
    timestep_schedule,
    noise_prediction_model,
    gaussian_channel_simulator,
    chunk_seeds_per_step,
    Dkl_per_step,
    seed,
):
    """Decodes a compressed image representation back into its latent space form.

    Args:
        latent_shape (torch.Size): Shape of the target latent tensor to be reconstructed.
        latent_device (torch.device): Device (CPU/GPU) where the computation will be performed.
        latent_dtype (torch.dtype): Data type of the latent tensor (e.g., torch.float32).
        timestep_schedule (List[float]): List of timesteps in decreasing order.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.
        gaussian_channel_simulator: Simulator used for gaussian channel reconstruction.
        chunk_seeds_per_step (List[List[int]]): Compressed representation of the image,
            consisting of lists of integer seeds for each denoising step.
        Dkl_per_step (List[float]): List of Kullback-Leibler divergence values per step,
            used to reconstruct the denoising process.
        seed (int): Random seed for reproducibility of the denoising process.

    Returns:
        torch.Tensor: The reconstructed latent representation of the image, obtained
            through progressive denoising steps guided by the compressed representation.
    """

    torch.manual_seed(seed)
    noisy_latent = torch.randn(latent_shape, device=latent_device, dtype=latent_dtype)
    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)
    for step_index, (prev_timestep, chunk_seeds, Dkl) in enumerate(
        zip(timestep_schedule, chunk_seeds_per_step, Dkl_per_step)
    ):
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep
        )
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)
        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr)
        sample = gaussian_channel_simulator.decode(
            chunk_seeds, noisy_latent.size(), Dkl, seed=step_index
        )
        reshaped_sample = (
            sample.reshape(latent_shape).to(latent_device).to(latent_dtype)
        )
        noisy_latent = reshaped_sample * std + p_mu
        current_snr = prev_snr

    return noisy_latent

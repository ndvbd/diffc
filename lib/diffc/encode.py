import torch
from lib.diffc.utils.p import P
from lib.diffc.utils.q import Q
from tqdm import tqdm


@torch.no_grad()
def encode(
    target_latent, # the real image latent vector
    timestep_schedule,
    noise_prediction_model,
    gaussian_channel_simulator,
    manual_dkl_per_step=None,
    recon_timesteps=[],
    seed=0,
):
    """Creates a compressed representation of an image using a diffusion model.

    Args:
        target_latent: Latent representation of the image to encode, as produced by the
            diffusion model's VAE encoder.
        timestep_schedule: List of timesteps, parallel to SNR_schedule. The timesteps should match the SNRs that the diffusion model expects at those timesteps.
        SNR_schedule: List of signal-to-noise ratios, decreasing towards zero (e.g.,
            [0.8, 0.6, 0.4, 0.2, 0.1]). SNR values must be in the set of values expected
            by the predict_noise function. Last element must be > 0. Ending with '0'
            (lossless compression of the latent) is not currently supported (and probably
            not desirable).
        predict_noise: Callable which takes in a noisy latent and that latent's SNR, and
            returns a prediction of the latent's noise component.
        gaussian_channel_simulator: Used for gaussian channel simulation.
        manual_dkl_per_step: Used to manually hard-code the dkl per step. Otherwise we'd
            need to send it as side-information. TODO: fancier entropy models of dkl per
            step?
        recon_timesteps: List of timesteps in decreasing order. When used, saves the noisy
            latents from the encoding process at each timestep.
        seed:
            random seed for the compression process.

    Returns:
        tuple:
            - chunk_seeds_per_step (List[List[int]]): One list of ints per step. This is
              the compressed representation of the image, although it still needs to be
              entropy coded. Fed back into the gaussian channel simulator for decoding.
            - dkl_per_step (List[float]): This is also fed back in to the gaussian
              channel simulator to reconstruct the denoising process.
            - noisy_recons: Noisy reconstructions of the target image generated during
              the encoding process. These will be the same noisy reconstructions
              generated during decoding. For faster evaluation, we can skip decoding and
              just use these recons.
            - noisy_recon_step_indices (List[float]): List which is parallel to
              noisy_recons, and reports the step index for each recon.
    """
    chunk_seeds_per_step = []
    dkl_per_step = []
    noisy_recons = []
    noisy_recon_step_indices = []
    recon_timesteps = recon_timesteps.copy()

    torch.manual_seed(seed) # seed is 0
    noisy_latent = torch.randn(
        target_latent.shape, device=target_latent.device, dtype=target_latent.dtype
    ) # generate noise

    current_timestep = 1000
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    for step_index, prev_timestep in tqdm(
        enumerate(timestep_schedule), total=len(timestep_schedule)
    ):  # "previous" as in closer to 1 than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            noisy_latent, current_timestep  # current_timestep starts from 1000
        )

        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep) # 0.08 for step 972

        p_mu, std = P(noisy_latent, noise_prediction, current_snr, prev_snr) # p_mu has shape of latent but std is scalar. if we are in step 1000, P() predicts the mean of x999. we dont see the real image here, so we can call it prior

        q_mu = Q(noisy_latent, target_latent, current_snr, prev_snr) # this line can be done only by sender (not recipient), and it will provide the guidance to choose the best prng index for noise selection. Q is just an interpolation between x0 (the real image = target latent) and x_t (noisy latent). since we see here the real image, we can call it posterior

        q_mu_flat_normed = ((q_mu - p_mu) / std).flatten().detach().cpu().numpy()  # convert latent to vector (flat). also normalize it so a normal-prng-noise will have chance to be close to it

        manual_dkl = (
            None if manual_dkl_per_step is None else manual_dkl_per_step[step_index]
        ) # the amount of information we want to send at this step

        sample, chunk_seeds, dkl = gaussian_channel_simulator.encode(
            q_mu_flat_normed, manual_dkl=manual_dkl, seed=step_index
        )  # find the noise-indices that generate noise that is closest to q_mu. i think that sample is the closest noise to q_mu_flat_normed using the indices in chunk_seeds
        chunk_seeds_per_step.append(chunk_seeds)
        dkl_per_step.append(dkl)
        sample = torch.tensor(sample)
        reshaped_sample = (
            sample.reshape(noisy_latent.shape)
            .to(noisy_latent.device)
            .to(noisy_latent.dtype)
        ) # back to shape 1,4,64,89 - the shape of the latent
        noisy_latent = reshaped_sample * std + p_mu  # we take our best prediction of previous step (also the decoder will have it) and continue the process from there
        current_timestep = prev_timestep
        current_snr = prev_snr

        ## Optionally, save the current reconstruction
        save_current_latent = False
        while len(recon_timesteps) > 0 and current_timestep <= recon_timesteps[0]:
            save_current_latent = True
            recon_timesteps = recon_timesteps[1:]

        if save_current_latent:
            noisy_recons.append(noisy_latent)
            noisy_recon_step_indices.append(step_index)

    return chunk_seeds_per_step, dkl_per_step, noisy_recons, noisy_recon_step_indices

import torch
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from lib.noise_prediction_models.latent_noise_prediction_model import LatentNoisePredictionModel

# TODO this is totally untested Claude code
class SDXLModel(LatentNoisePredictionModel):
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0", use_refiner=False, device="cuda", dtype=torch.float16):
        """
        Initialize the SDXL model.
        
        Args:
            model_id (str): HuggingFace model ID for SDXL
            device (str): Device to run the model on ("cuda" or "cpu")
            dtype (torch.dtype): Data type for model parameters
        """
        self.use_refiner = use_refiner
        self.device = device
        self.dtype = dtype
        
        # Initialize the pipeline
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            vae=self.vae
        )
        self.pipe = self.pipe.to(self.device)
        
        # Store key components for easy access

        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2

        if self.use_refiner:
            # Load just the UNet and text encoder 2 from the refiner model
            self.refiner_unet = UNet2DConditionModel.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                subfolder="unet",
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            
            # The refiner doesn't use text_encoder_1, only text_encoder_2 and its projections
            self.refiner_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0", 
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)

        
        # Initialize DDPM scheduler with 1000 steps for SNR calculations
        self.reference_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.reference_scheduler.set_timesteps(num_inference_steps=1000)
        
        # Pre-compute alphas, timesteps, and SNR values for the scheduler
        self.timesteps = self.reference_scheduler.timesteps.to(device)
        alphas = self.reference_scheduler.alphas_cumprod.to(device)
        self.snr_values = torch.sqrt(alphas / (1 - alphas))
        
        # Initialize configuration attributes
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.add_text_embeds = None
        self.negative_add_text_embeds = None
        self.add_time_ids = None
        self.guidance_scale = None
        self.image_width = None
        self.image_height = None

    def get_timestep_snr(self, timestep):
        """
        Get the signal-to-noise ratio for a given timestep.
        
        Args:
            timestep (int): Timestep index
            
        Returns:
            torch.Tensor: SNR value for the timestep
        """
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep-1]

    def image_to_latent(self, img_pt):
        """
        Convert input image tensor to latent representation.
        
        Args:
            img_pt (torch.Tensor): Image tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            torch.Tensor: Latent representation
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)
        
        # Move input to correct device and type
        img_pt = img_pt.to(device=self.device, dtype=self.dtype) * 2 - 1
        
        # Get VAE scaling factor
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        # Encode image to latent space
        latent = self.vae.encode(img_pt).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        
        return latent

    def latent_to_image(self, latent):
        """
        Convert latent representation back to image.
        
        Args:
            latent (torch.Tensor): Latent tensor
            
        Returns:
            torch.Tensor: Decoded image in range [-1, 1]
        """
        # Scale latents
        latent = latent / self.vae.config.scaling_factor
        
        # Decode to image space
        with torch.no_grad():
            image = self.vae.decode(latent).sample
            
        return (image / 2 + 0.5).clamp(0, 1).detach()

    def configure(self, prompt, prompt_guidance, image_width, image_height):
        """
        Configure model with prompt and parameters.
        
        Args:
            prompt (str or List[str]): Text prompt(s)
            prompt_guidance (float): Classifier-free guidance scale
            image_width (int): Output image width
            image_height (int): Output image height
        """
        self.guidance_scale = prompt_guidance
        self.image_width = image_width
        self.image_height = image_height
        
        # Get original and target sizes for SDXL conditioning
        original_size = (image_height, image_width)
        target_size = (image_height, image_width)
        crops_coords_top_left = (0, 0)
        
        # Process the prompt through both text encoders
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        
        # Get prompt embeddings from both encoders
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder(text_input_ids)[0]
            prompt_embeds_2 = self.text_encoder_2(text_input_ids_2, output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds_2[0]
            prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
            
        # Concatenate embeddings from both text encoders
        self.prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        self.add_text_embeds = pooled_prompt_embeds
        
        # Handle zero-guidance case
        if prompt_guidance > 1.0:
            # Process empty prompt for classifier-free guidance
            uncond_tokens = [""] * (len([prompt]) if isinstance(prompt, str) else len(prompt))
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids.to(self.device)
            
            uncond_input_2 = self.tokenizer_2(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids_2 = uncond_input_2.input_ids.to(self.device)
            
            with torch.no_grad():
                negative_prompt_embeds_1 = self.text_encoder(uncond_input_ids)[0]
                negative_prompt_embeds_2 = self.text_encoder_2(uncond_input_ids_2, output_hidden_states=True)
                negative_pooled_prompt_embeds = negative_prompt_embeds_2[0]
                negative_prompt_embeds_2 = negative_prompt_embeds_2.hidden_states[-2]
                
            self.negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)
            self.negative_add_text_embeds = negative_pooled_prompt_embeds
        
            # Prepare time embeddings
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size
            )
            self.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            
        else:
            self.negative_prompt_embeds = None
            self.negative_add_text_embeds = None
            self.add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size
            )

    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise in the latent at given timestep.
        
        Args:
            noisy_latent (torch.Tensor): Noisy latent tensor
            timestep (int): Timestep value
            
        Returns:
            torch.Tensor: Predicted noise
        """
        # Select appropriate UNet based on timestep
        unet = self.refiner_unet if self.use_refiner and timestep <= 200 else self.unet

        # Handle classifier-free guidance
        if self.guidance_scale > 1.0:
            # Duplicate input for conditional and unconditional paths
            latent_model_input = torch.cat([noisy_latent] * 2).to(self.device).to(self.dtype)
            
            # Prepare condition inputs
            prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])
            add_text_embeds = torch.cat([self.negative_add_text_embeds, self.add_text_embeds])
            
            # Get both unconditional and conditional predictions
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": self.add_time_ids
                }
                noise_pred = unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
            
            # Apply guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        else:
            # Direct prediction without guidance
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": self.add_text_embeds,
                    "time_ids": self.add_time_ids
                }
                noise_pred = unet(
                    noisy_latent,
                    timestep,
                    encoder_hidden_states=self.prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
        
        return noise_pred

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size):
        """
        Helper method to prepare the additional time embeddings used by SDXL.
        
        Args:
            original_size (tuple): Original image size (H, W)
            crops_coords_top_left (tuple): Crop coordinates (x, y)
            target_size (tuple): Target image size (H, W)
            
        Returns:
            torch.Tensor: Additional time embeddings
        """
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.prompt_embeds.dtype)
        return add_time_ids.to(self.device)
# demystifying-sd-finetuning

## Outline

 - Setup: training curves are noisy, how can we identify any meaningful trends?
   - Background on how diffusion models work, sources of randomness
     - inference: random noise
	 - training: random noise, random timestep
 - Analyze loss vs timestep (graph)
 - Introduce stable loss
   - Constant seed (credit to [/u/fpgaminer](https://old.reddit.com/r/StableDiffusion/comments/1gdkpqp/the_gory_details_of_finetuning_sdxl_for_40m/))
   - SNR compensation
 - Introduce train loss vs test/val loss
   - General ML principles, find external explanation
   - Show the interaction across a training run on stable losses
 - Demonstrate how loss @ training checkpoints affects generated images, accuracy vs variety tradeoff
 - LR sweep to find optimal steps x lr
   - Concept: total learning effort?
   - Should LR scale by batch_size or sqrt(batch_size)? Seems to be sqrt()
 - Test different dataset sizes
 - Test optimizations
 - Random crop?
 - Compare finetune vs lora
 - Call to action: implement stable training loss and validation loss in popular trainer tools (kohya, onetrainer, etc)

## Introduction

Finetuning large pretrained diffusion models like Stable Diffusion (or rf models like Flux, SD3, etc) can be diffucult, requiring a large amount of trial and error to get right. There are probably a thousand different videos and articles giving advice and tips by now, but most of them fit into the category of "here's what I found through trial and error, now copy my settings". This is hopefully different. My aim here is to explain why it's such a hard problem to find clear answers, and how with the appropriate tools, you can find those answers for yourself much more easily. I'm going to use stable diffusion 1.5 for all the experiments here, but that's just for the sake of speed, and the same principles will apply to any diffusion or flow model.

## Training loss curves are noisy

If you've finetuned SD or trained a lora before, you've probably spent some time staring at something like this:

![image](https://github.com/user-attachments/assets/e0b8d6ac-8e6b-4072-8bf8-1e2263e8e43c)

...and trying to look for meaning in it. Does it drop slightly over time, or is that just random movement? SHOULD you be able to see a pattern?

(spoiler: probably not, in most cases)

In order to understand why this is so noisy, first we need to look at the training process itself. This is a simplified version of how sd1.5 is called during training:

```python
latents = vae_encode(pixels)
encoder_hidden_states = te_encode(captions)
timesteps = torch.randint(low=0, high=1000, (batch_size,))
noise = torch.randn_like(latents)
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
model_pred = unet(noisy_latents, timesteps, encoder_hidden_states)
loss = mse_loss(model_pred, noise)
```

There are two places where a random function is called. We sample a <ins>random timestep</ins> from the range of the model's noise schedule, and we sample <ins>random noise</ins> in the same shape as the encoded latent image. The noise is added to the latent, using the timestep to determine the strengh of the noise. We can think of the timestep as controling the *signal to noise ratio* (SNR) of the noisy_latents input, and in fact there's a [handy little utility in diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L60) that lets you calculate the SNR for any given timesteps.

What if, just as a test (NOT training the model), we sample each timestep in order (still using random noise though!), and graph the SNR and loss across the whole noise schedule?

![image](https://github.com/user-attachments/assets/492591d0-7cea-4b2c-922d-ec6e7c3c847a)

Note that the Y axis is in logarithmic scale. At low timesteps, where the input is mostly image (high SNR), the model struggles to estimate the noise, with loss values nearing 1. At high timesteps, the SNR is low and the model input is mostly noise, which makes it easy for it to estimate the noise, and loss values are as low as 0.002

Isn't it interesting though, that the loss corresponds so smoothly with the timestep? There's still some noise, particularly towards the low SNR end, but overall it's very clean and predictable, especially considering that this still using *different random noise for each sample.* This tells us that most of the noise in our training loss curve is actually coming from the random timesteps, not the random latent noise.

We can fit a function to this and use it to flatten out the curve, so that the loss values are more similar across the schedule. The best fit I found in this case was an exponential function, and it's not perfect, but it does at least reduce the range somewhat:

![image](https://github.com/user-attachments/assets/20bd49bc-c302-45f1-93f0-007ee994e3e9)

Close enough at least that the range fits in one order of magnitude now, and is reasonably close to a normal distribution, instead of the wild exponential distribution we had before. If I log this debiased loss during an actual training run, it's still noisy, but it does play a little bit nicer with the smoothing filter in tensorboard:

![image](https://github.com/user-attachments/assets/2b2f186e-6a95-44b3-8796-23ef8094949f)

It's possible that a better equation could fit the loss curve better and cancel out more of the timestep noise, or we could potentially train a tiny neural network on the fly to fit it near perfectly. Ultimately though, it doesn't really matter that much, because this is only for visualization, so it won't help the model learn faster or anything cool like that.

## Stable loss

So, we can take some of the noise out of the training loss curve, but

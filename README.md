# demystifying-sd-finetuning

## Outline

 - [x] Setup: training curves are noisy, how can we identify any meaningful trends?
   - [x] Background on how diffusion models work, sources of randomness
     - [ ] inference: random noise
     - [x] training: random noise, random timestep
 - [x] Analyze loss vs timestep (graph)
 - [x] Introduce stable loss
   - [x] Constant seed
   - [x] SNR compensation
 - [x] Introduce train loss vs test/val loss
   - [ ] General ML principles, find external explanation
   - [x] Show the interaction across a training run on stable losses
 - [ ] Demonstrate how loss @ training checkpoints affects generated images, accuracy vs variety tradeoff
 - [x] LR sweep to find optimal steps x lr
   - [ ] Concept: total learning effort?
   - [x] Should LR scale by batch_size or sqrt(batch_size)? Seems to be sqrt()
 - [ ] Test different dataset sizes
 - [ ] Test optimizations
 - [ ] Random crop?
 - [ ] Compare finetune vs lora
 - [ ] Call to action: implement stable training loss and validation loss in popular trainer tools (kohya, onetrainer, etc)

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

So, we can take some of the noise out of the training loss curve, but what if we go one level further? We could periodically evalute some fixed subset of images, using a fixed seed for any random functions, so that the same timesteps and the same noise are used every time. Credit to [/u/fpgaminer](https://old.reddit.com/r/StableDiffusion/comments/1gdkpqp/the_gory_details_of_finetuning_sdxl_for_40m/) for some of the code in my implementation. It's not a completely new idea, but I think it deserves more attention, because it's invaluable for monitoring training progress.

How it works: every N steps, we pause training, store the random states, and manually seed to a fixed value. We can evaluate some number of images, average the loss across all of them, log it, then restore the old random states and continue training. See train_sd.py for specifics. I also went a bit further and added timestep buckets (for more uniform distribution across the noise schedule) and debiased loss, to make the average as clean as possible. Now we have a deterministic loss value that gets logged over the training run, which we can use to judge how the model is learning:

![image](https://github.com/user-attachments/assets/6372ab64-97f8-4a2d-837b-7ac40d05f920)

This is what that stable loss curve looks like, evaluated on 2 images in the training dataset. It's nice and clean, which is great, but it's basically a straight line, so what's going on? Is it just continuously improving? Should we keep going for more steps to find where it bottoms out?

## Dataset splits

There's a very important principle in machine learning: <ins>Split your dataset into training and validation subsets.</ins> Large neural networks can memorize a lot of data, so in order to evalute whether it's learning good general knowledge, or simply memorizing the specifics of the training data, you need to hold some portion of the data out from the training dataset.

To start with, I'm working with a very small dataset of 22 images, so I will just manually create my splits. I'm keeping 20 images in the training split, and holding out 2 for validation, and hand picking those two to be somewhat representative of the whole set. This is an extremely small dataset by ML standards, but it's a decent example for personalization training. Based on my observations, most people use something in the range of 5-50 images for training an identity model/lora. I think 10% is generally considered a good validation split, but you could use more or less depending on your dataset size. For million+ scale datasets, sampling 100,000 validation images would just be wasteful and unnecessary.

Here's that same training run again, but this time with two stable loss curves - one for training images, and one for validation images:

![image](https://github.com/user-attachments/assets/156437e4-0f3b-49e8-b873-0cb1eec80367)

See that U-shaped curve for validation? That's exactly what we're looking for. The decreasing validation loss means that the model is learning patterns that *generalize to images it has not seen before.* (Not trained on, anyway. Obviously we're evaluating the model on them, but it's not allowed to optimize over them.) As the curve flattens out and starts to rise again, that's where the model is learning patterns that are specific to the images in the training dataset, and don't generalize to the validation set. AKA, overtraining. Note how by the end of this run, the validation loss is actually worse than at the start.

But...do we actually want to stop where validation loss is the lowest? If we were training a model from scratch on a huge dataset, probably. In this case where the goal is to generate images that look like a specific person, maybe that's not ideal. It's possible that overtraining by some amount could actually improve the quality of generated images, at the expense of some flexibility. But at least this gives us some way to see exactly when we're undertraining or overtraining, and it will be very helpful when we're adjusting hyperparameters like the learning rate.

# ADD SAMPLE IMAGES

## Learning Rate sweep

Lets's run a sweep of different learning rates, and compare the validation curves.

![image](https://github.com/user-attachments/assets/906d9919-2406-4f99-b542-2e17aca74d0e)

In order from highest to lowest LR, this is [5e-6, 1e-6, 5e-7, 3e-7, 2e-7, 1e-7]. We can see that learning rate does, as expected, have a mostly linear effect on the speed of convergence. More interesting though, it looks like all the runs reached a similar minimum validation loss, although the higher LR curves are more noisy. If you push the LR too high, it could eventually cause training instability, but below that, there seems to be little benefit to going lower unless you need more steps to get through a large dataset. Personally, I would not have expected that the lowest validation loss across these runs would come from the highest tested learning rate after only `20 images * 20 epochs = 400 steps`.

## Scaling LR vs Batch Size

The common advice I've seen is that when changing the batch size, you should scale the learning rate linearly with it. This behavior is written into every diffusers example training script, and many other training tools. Older theory stated that you should scale learning rate by the square root of the batch size to keep the variance constant, but about a decade ago it was found that for large batch sizes, it's better to scale linearly (https://arxiv.org/abs/1404.5997).

I'm scaling the logged step by the batch size, so that the curves line up based on the number of images sampled, for more convenient viewing.

Here's a series of runs with learning_rate * batch_size, for batch size in [1, 2, 4]:

![image](https://github.com/user-attachments/assets/00695a74-6125-4859-a7f8-bb6820bc1299)

And here's learning_rate * sqrt(batch_size):

![image](https://github.com/user-attachments/assets/57c84818-9277-4487-accc-e03961f516ed)

Looks like in this case at least, with a UNet model and small batch sizes, the square root rule is correct. It's possible that DiT models might behave differently? Further testing is required. I don't know at what point you would start needing to scale linearly, but if you're training on a single GPU, square root is probably the way to go, and if you're using large enough batch sizes to start worrying about this, you should probably be running your own tests instead of blindly following some random guy on the internet :)

[From what I can find](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md), Stable Diffusion 1.5 was originally trained at a batch size of 2048 and a learning rate of 1e-4. If we divide `1e-4 / sqrt(2048)`, we get a base LR of 2.2e-6 for a batch size of 1, which seems reasonable based on the results of the learning rate sweep. If we instead use linear scaling and divide `1e-4 / 2048`, that would give us a base LR of 4.9e-8, which just seems unnecessarily low.

This also shows that the batch size doesn't have much effect on the quality of the result when learning rate is scaled appropriately. I had expected that larger batch sizes might generalize better, but it seems that AdamW does just fine even at `batch_size = 1`. The main benefit of larger batch sizes then, is more efficient hardware utilization:

| batch size | steps |  time |      speed     |
| ---------- | ------| ----- | -------------- |
|      1     |  5000 | 34:23 | 2.4 images/sec |
|      2     |  2500 | 27:44 | 3.0 images/sec |
|      4     |  1250 | 23:54 | 3.5 images/sec |

## Text encoder training

Is it a good idea to train the text encoder(s)? Is it bad? Does it even matter? All of the major image diffusion models that I'm aware of use unmodified pretrained text encoders, rather than finetuning them in the context of image generation. Some community members believe that it can help, if done right, but it also seems that using too high a learning rate for the text encoder, or training it for too long, can cause it to collapse in some way. A common rule is to either use a lower learning rate for the text encoder compared to the diffusion model, or else stop training it after some number of steps. I tested training the text encoder for all steps, at different multipliers [1x, 0.5x, 0.25x] of the unet learning rate.

![image](https://github.com/user-attachments/assets/be3fcb34-84d2-49b2-a0c0-3506fb2d4627)

![image](https://github.com/user-attachments/assets/678d9ca8-1d24-486e-9a00-1675de0cfec7)

Very close results, although it does move the curves to the left slightly, probably just due to the increased number of trainable parameters. The minimum loss doesn't change meaningfully though, similar to with the learning rate sweeps. One way to think of it: if we consider the unet-only case as an example of `unet @ lr` and `te @ (lr * 0)`, we're effectively just increasing the average learning rate over the combination of both models. That left/right shift could lead people to falsely believe that enabling/disabling text encoder training helped or hurt, when really it just changed the number of training steps required for an equivalent result.

Worth noting that the captions for this dataset are probably in distribution for CLIP, so your results may vary. Most concepts will be in distribution already though, to be fair, since CLIP was trained on internet scale data. ([From the paper](https://arxiv.org/abs/2103.00020), they collect 400M image/text pairs based on searching for any word that appears in english wikipedia at least 100 times (500k total concepts), and approximately class balance by limiting each query to 20k results. No other forms of filtering.)

I suspect that if you want to add concepts that are actually new, not just new instances of an existing concept, it might be better to finetune CLIP with its original image encoder and contrastive loss, as that would allow it to actually learn the new concepts in their proper context rather than just treating the text encoder like an embedding layer. [Few people have even attempted real CLIP training though.](https://huggingface.co/zer0int)

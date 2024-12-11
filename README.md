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
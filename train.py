import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import Unet, GaussianDiffusion, Trainer


if __name__ == "__main__":
    model = Unet(
        channels=3,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        timesteps=1000,  # number of steps
        sampling_timesteps=20,
        training_stage='stage2'
    )

    trainer = Trainer(
        diffusion,
        data_dir='/data/SIED_dataset',
        train_dataset='train_sony_0.01',
        val_dataset='test_sony_0.01',
        results_folder='./results_sony_0.01',
        ckpt_path='ckpt/stage1_sony/0.01',
        patch_size=[512, 512],
        train_batch_size=1,
        train_lr=1e-4,
        save_and_sample_every=5000,
        train_num_steps=400000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False
    )

    # trainer.load('stage1_weight') # if training_stage='stage2'
    trainer.train()
    
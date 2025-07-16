import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from models import Unet, GaussianDiffusion
from pathlib import Path
import torch
from dataset.dataloader import Test_Dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from metric import get_ssim_torch, get_psnr_torch
import lpips


def data_transform(X):
    return 2 * X - 1.0


def tensor_gpu(batch, check_on=True):
    def check_on_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            tensor_g = tensor_
        else:
            tensor_g = tensor_.cuda()
        return tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            return tensor_

        if tensor_.is_cuda:
            tensor_c = tensor_.cpu()
        else:
            tensor_c = tensor_
        tensor_c = tensor_c.detach().numpy()
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            for k, v in batch.items():
                batch[k] = check_on_gpu(v)
        else:
            for k, v in batch.items():
                batch[k] = check_off_gpu(v)
    else:
        if check_on:
            batch = batch
        else:
            for k, v in batch.items():
                batch[k] = v.detach().numpy()

    return batch


class inference:
    def __init__(
            self,
            diffusion_model,
            data_dir,
            val_dataset,
            *,
            batch_size=128,
            ema_update_every=10,
            ema_decay=0.995,
            results_folder='./results',
            ckpt_path='ckpt/stage2',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
    ):
        super().__init__()

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[ddp_kwargs]
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.batch_size = batch_size
        self.dir = data_dir
        self.va_dataset = val_dataset

        self.ckpt_path = ckpt_path

        self.val_ds = Test_Dataset(image_dir=self.dir, 
                                   filelist='{}.txt'.format(self.va_dataset))

        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                            pin_memory=True, num_workers=8)

        self.val_dl = val_dl

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

    @property
    def device(self):
        return self.accelerator.device

    def load(self):

        device = self.accelerator.device
        data = torch.load(os.path.join(self.ckpt_path), map_location=device)

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

    def infer(self):

        self.load()
        
        self.ema.ema_model.eval()
        
        loss_fn_alex = lpips.LPIPS(net='alex').eval().cuda()
        
        psnr, ssim, Lpips = 0, 0, 0
        with torch.inference_mode():
            for i, data_batch in enumerate(self.val_dl):
                data_batch = tensor_gpu(data_batch)
                
                img_name = data_batch["img_name"][-1] 
                
                pred_img, high_img = self.ema.ema_model.sample(data_batch, return_all_timesteps=False)
                
                Lpips += loss_fn_alex(data_transform(pred_img), data_transform(high_img)).detach().cpu().numpy().item()
                
                pred_img = torch.clip(pred_img * 255.0, 0, 255.0)
                high_img = torch.clip(high_img * 255.0, 0, 255.0)

                psnr += get_psnr_torch(pred_img, high_img).item()
                ssim += get_ssim_torch(pred_img, high_img).item()
                
                pred_img = pred_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
                pred_img = Image.fromarray(pred_img)
                pred_img.save(os.path.join(self.results_folder, img_name))
                
                print('processing img:{}'.format(img_name))
                
            avg_psnr, avg_ssim, avg_lpips = psnr / len(self.val_dl), ssim / len(self.val_dl), Lpips / len(self.val_dl)
            
            print('avg_psnr:{:.4f}, avg_ssim:{:.4f}, avg_lpips:{:.4f}'.format(avg_psnr, avg_ssim, avg_lpips))
            
        return avg_psnr, avg_ssim, avg_lpips


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

    inference = inference(
        diffusion,
        data_dir='/data/SIED_dataset',
        val_dataset='test_canon_0.01',
        results_folder='./infer_canon_0.01',
        ckpt_path='./ckpt/canon/canon_0.01.pt',
        batch_size=1
    )
    
    PSNR, SSIM, LPIPS = [], [], []
    
    for i in range(3):

        avg_psnr, avg_ssim, avg_lpips = inference.infer()
        PSNR.append(avg_psnr)
        SSIM.append(avg_ssim)
        LPIPS.append(avg_lpips)
        
    print(PSNR)
    print(SSIM)
    print(LPIPS)
    print('mean_psnr:{:.4f}, mean_ssim:{:.4f}, mean_lpips:{:.4f}'.format(sum(PSNR) / len(PSNR), sum(SSIM) / len(SSIM), sum(LPIPS) / len(LPIPS)))
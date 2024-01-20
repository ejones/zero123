import sys
import time
import fileinput
import json
import os.path
import uuid
import traceback
import math

import numpy as np
import torch
from contextlib import nullcontext
import diffusers  # 0.12.1
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms

from zero123.ldm.models.diffusion.ddim import DDIMSampler
from zero123.ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config


_GPU_INDEX = 0


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z, img_callback=None):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]

            if img_callback:
                def img_callback_sampler(pred_x0, i):
                    def decode_img():
                        pred_x0_ddim = model.decode_first_stage(pred_x0)
                        return torch.clamp((pred_x0_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                    img_callback(decode_img, i)
            else:
                img_callback_sampler = None

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None,
                                             img_callback=img_callback_sampler)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(carvekit_model, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(carvekit_model, input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    #print('new input_im:', lo(input_im))

    return input_im


class ImageViewpointGenerator:
    def __init__(self, device, ckpt_path, config_path):
        self.device = device

        config = OmegaConf.load(config_path)

        print('Instantiating LatentDiffusion...')
        self.turncam_model = load_model_from_config(config, ckpt_path, device=device)

        print('Instantiating Carvekit HiInterface...')
        self.carvekit_model = create_carvekit_interface()

    def preprocess(self, image):
        im_data = load_and_preprocess(self.carvekit_model, image)
        return Image.fromarray(im_data.astype(np.uint8))

    def generate(self, params):
        raw_im = params['image']
        x = params.get("x", 0.0)
        y = params.get("y", 0.0)
        z = params.get("z", 0.0)
        preprocess = params.get("preprocess", True)
        scale = params.get("scale", 3.0)
        n_samples = params.get("n_samples", 4)
        ddim_steps = params.get("ddim_steps", 50)
        ddim_eta = params.get("ddim_eta", 1.0)
        precision = params.get("precision", 'fp32')
        h = params.get("h", 256)
        w = params.get("w", 256)
        img_callback = params.get('img_callback')

        raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
        input_im = preprocess_image(self.carvekit_model, raw_im, preprocess)

        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(self.device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        if img_callback:
            def sample_img_callback(decode_samples, i):
                def get_imgs():
                    samples = decode_samples() 
                    ims = []
                    for sample in samples:
                        sample = 255.0 * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                        ims.append(Image.fromarray(sample.astype(np.uint8)))
                    return ims
                img_callback(get_imgs, i)
        else:
            sample_img_callback = None

        sampler = DDIMSampler(self.turncam_model, device=self.device)
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(input_im, self.turncam_model, sampler, precision, h, w,
                                      ddim_steps, n_samples, scale, ddim_eta, used_x, y, z,
                                      img_callback=sample_img_callback)

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        return output_ims


def main(
        device_idx=_GPU_INDEX,
        ckpt='105000.ckpt',
        config='configs/sd-objaverse-finetune-c_concat-256.yaml'):

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1 and sys.argv[1][:1].isalpha():
        # HACK
        device = sys.argv[1]
    else:
        if len(sys.argv) > 1:
            print('old device_idx:', device_idx)
            device_idx = int(sys.argv[1])
            print('new device_idx:', device_idx)

        device = f'cuda:{device_idx}'

    print('device:', device)

    viewpoint_gen = ImageViewpointGenerator(device, ckpt, config)

    for line in fileinput.input(('-',)):
        try:
            params = json.loads(line)
            out_dir = params.pop('outdir')
            params['image'] = Image.open(params['image'])

            output_ims = viewpoint_gen.generate(params)

            output_im_paths = []
            for output_im in output_ims:
                out_name = f'{uuid.uuid4()}.png'
                output_im.save(os.path.join(out_dir, out_name))
                output_im_paths.append(out_name)

            print(json.dumps(output_im_paths))
        except:
            traceback.print_exc()


if __name__ == '__main__':
    main(*sys.argv[1:])

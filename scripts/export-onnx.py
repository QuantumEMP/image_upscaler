"""
Converts the official Real-ESRGAN PyTorch weights to ONNX for onnxruntime-web,
splits them into chunks under Cloudflare's 25 MiB static-asset limit and writes
public/models/manifest.json.

    python3 -m venv .venv && .venv/bin/pip install torch onnx onnxscript
    .venv/bin/python scripts/export-onnx.py

Architectures are copied from basicsr (RRDBNet) and Real-ESRGAN (SRVGGNetCompact).
"""
import json
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

RELEASES = 'https://github.com/xinntao/Real-ESRGAN/releases/download'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'public', 'models')
CACHE_DIR = os.path.join(ROOT, '.cache', 'weights')
CHUNK_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------- RRDBNet
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_block, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# -------------------------------------------------------- SRVGGNetCompact
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_conv, num_feat=64, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=float(self.upscale), mode='nearest')


MODELS = [
    dict(id='realesr-general-x4v3', label='General (fast)', description='Compact model for photos. Fast, good default.',
         url=f'{RELEASES}/v0.2.5.0/realesr-general-x4v3.pth', build=lambda: SRVGGNetCompact(num_conv=32), tile=256),
    dict(id='realesrgan-x4plus', label='General (best)', description='Full RRDB model. Highest quality for photos, much slower.',
         url=f'{RELEASES}/v0.1.0/RealESRGAN_x4plus.pth', build=lambda: RRDBNet(num_block=23), tile=128),
    dict(id='realesrgan-x4plus-anime', label='Anime / illustration', description='Optimised for anime art and drawings.',
         url=f'{RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth', build=lambda: RRDBNet(num_block=6), tile=192),
    dict(id='realesr-animevideov3', label='Anime (fast)', description='Tiny, very fast model for anime-style images.',
         url=f'{RELEASES}/v0.2.5.0/realesr-animevideov3.pth', build=lambda: SRVGGNetCompact(num_conv=16), tile=256),
]


def load_weights(url):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, os.path.basename(url))
    if not os.path.exists(path):
        print(f'  downloading {url}')
        urllib.request.urlretrieve(url, path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('params_ema', 'params'):
        if key in state:
            return state[key]
    return state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for m in MODELS:
        print(f"Exporting {m['id']}")
        net = m['build']()
        net.load_state_dict(load_weights(m['url']), strict=True)
        net.eval()

        onnx_path = os.path.join(OUT_DIR, f"{m['id']}.onnx")
        torch.onnx.export(
            net, torch.rand(1, 3, 64, 64), onnx_path,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'h4', 3: 'w4'}},
            opset_version=17, dynamo=False,
        )

        # Split into chunks below the 25 MiB per-file limit of Cloudflare static assets.
        data = open(onnx_path, 'rb').read()
        os.remove(onnx_path)
        for old in os.listdir(OUT_DIR):
            if old.startswith(m['id'] + '.onnx.part'):
                os.remove(os.path.join(OUT_DIR, old))
        parts = []
        for i in range(math.ceil(len(data) / CHUNK_BYTES)):
            name = f"{m['id']}.onnx.part{i}"
            with open(os.path.join(OUT_DIR, name), 'wb') as f:
                f.write(data[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES])
            parts.append(name)

        manifest.append({
            'id': m['id'], 'label': m['label'], 'description': m['description'],
            'scale': 4, 'tile': m['tile'], 'bytes': len(data), 'parts': parts,
        })
        print(f"  {len(data) / 1e6:.1f} MB in {len(parts)} part(s)")

    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump({'version': 1, 'models': manifest}, f, indent=2)


if __name__ == '__main__':
    main()

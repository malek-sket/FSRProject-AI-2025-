import torch
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as ssim

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=48, s=12, m=4):
        super(FSRCNN, self).__init__()
        # 1. Feature Extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )
        # 2. Shrinking, Mapping, Expanding
        layers = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            layers.extend([nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)])
        layers.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*layers)
        
        # 3. Deconvolution
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, 
                                            stride=scale_factor, padding=4, 
                                            output_padding=scale_factor-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        # تحسين الأجهزة الضعيفة: استخدام Residual Learning
        # نقوم بتكبير الصورة مسبقاً بطريقة Bicubic كقاعدة أساسية
        identity = torch.nn.functional.interpolate(x, scale_factor=self.last_part.stride[0], mode='bicubic', align_corners=False)
        
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        
        # النتيجة هي مجموع "التكبير التقليدي" + "تصحيح الذكاء الاصطناعي"
        return x + identity

# دوال المعالجة
def rgb_to_ycbcr(img):
    img = np.array(img).astype(np.float32)
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    return np.dstack((y, cb, cr))

def ycbcr_to_rgb(y, cb, cr):
    ycbcr = np.dstack((y, cb, cr))
    r = 298.082 * ycbcr[..., 0] / 256. + 408.583 * ycbcr[..., 2] / 256. - 222.921
    g = 298.082 * ycbcr[..., 0] / 256. - 100.291 * ycbcr[..., 1] / 256. - 208.120 * ycbcr[..., 2] / 256. + 135.576
    b = 298.082 * ycbcr[..., 0] / 256. + 516.412 * ycbcr[..., 1] / 256. - 276.836
    return np.clip(np.dstack((r, g, b)), 0, 255).astype(np.uint8)

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# def calc_ssim(img1, img2):
#     # تحويل التنسورات إلى numpy إذا لزم الأمر
#     img1 = img1.cpu().numpy().squeeze()
#     img2 = img2.cpu().numpy().squeeze()
#     # الحساب مع تحديد نطاق البيانات (0-1)
#     return ssim(img1, img2, data_range=1.0)

def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 64  # تغطية 128 بكسل في الصورة الأصلية
        elif args.scale == 3:
            args.patch_size = 42  # تغطية 126 بكسل في الصورة الأصلية
        elif args.scale == 4:
            args.patch_size = 32  # تغطية 128 بكسل في الصورة الأصلية
        return func(args)
    return wrapper

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
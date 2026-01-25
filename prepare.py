import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from tqdm import tqdm
import os
import functools
import concurrent.futures
from core import calc_patch_size, rgb_to_ycbcr

def process_single_image(image_path, args):
    """تعالج صورة واحدة وتعيد قائمة من الـ patches"""
    local_lr = []
    local_hr = []
    try:
        hr_pil = pil_image.open(image_path).convert('RGB')
        hr_y = rgb_to_ycbcr(hr_pil)[..., 0]

        w = (hr_y.shape[1] // args.scale) * args.scale
        h = (hr_y.shape[0] // args.scale) * args.scale
        hr_y = hr_y[:h, :w]
        
        hr_pil_y = pil_image.fromarray(hr_y.astype(np.uint8))
        lr_pil_y = hr_pil_y.resize((w // args.scale, h // args.scale), resample=pil_image.BICUBIC)
        
        hr_y_raw = np.array(hr_pil_y).astype(np.float32) / 255.0
        lr_y_raw = np.array(lr_pil_y).astype(np.float32) / 255.0

        for i in range(0, h - args.patch_size * args.scale + 1, args.stride * args.scale):
            for j in range(0, w - args.patch_size * args.scale + 1, args.stride * args.scale):
                hr_patch = hr_y_raw[i:i + args.patch_size * args.scale, j:j + args.patch_size * args.scale]
                lr_i, lr_j = i // args.scale, j // args.scale
                lr_patch = lr_y_raw[lr_i:lr_i + args.patch_size, lr_j:lr_j + args.patch_size]
                
                local_lr.append(lr_patch)
                local_hr.append(hr_patch)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        
    return local_lr, local_hr

@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, 'w', libver='latest')
    
    lr_dataset = h5_file.create_dataset('lr', shape=(0, args.patch_size, args.patch_size),
                                        maxshape=(None, args.patch_size, args.patch_size),
                                        chunks=(1, args.patch_size, args.patch_size), dtype=np.float32)
    hr_dataset = h5_file.create_dataset('hr', shape=(0, args.patch_size * args.scale, args.patch_size * args.scale),
                                        maxshape=(None, args.patch_size * args.scale, args.patch_size * args.scale),
                                        chunks=(1, args.patch_size * args.scale, args.patch_size * args.scale), dtype=np.float32)

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, '*')))
    
    temp_lr, temp_hr = [], []
    num_workers = max(1, os.cpu_count() - 6)
    print(f" بدء التجهيز بنظام التدفق على {num_workers} أنوية...")

    process_func = functools.partial(process_single_image, args=args)
    
    # السر هنا: لا نستخدم list() حول الـ executor.map
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # imap أو map هنا تعمل كـ generator، تعطينا النتائج أولاً بأول
        for l_lr, l_hr in tqdm(executor.map(process_func, image_paths), total=len(image_paths), desc="Streaming to Disk"):
            if l_lr:
                temp_lr.extend(l_lr)
                temp_hr.extend(l_hr)
            
            # الكتابة في الملف كلما جمعنا 2000 قطعة (حجم صغير لضمان انسيابية الرام)
            if len(temp_lr) >= 2000:
                write_to_h5(lr_dataset, hr_dataset, temp_lr, temp_hr)
                temp_lr, temp_hr = [], []
                # إجبار الهاردسك على استلام البيانات الآن وعدم تأجيلها
                h5_file.flush() 

    # كتابة أي بيانات متبقية في النهاية
    if len(temp_lr) > 0:
        write_to_h5(lr_dataset, hr_dataset, temp_lr, temp_hr)

    h5_file.close()
    print(f"\n✅ انتهى التجهيز! الرام ظلت مرتاحة والـ SSD عمل بانتظام.")

def write_to_h5(lr_ds, hr_ds, lr_list, hr_list):
    curr_idx = lr_ds.shape[0]
    add_count = len(lr_list)
    lr_ds.resize(curr_idx + add_count, axis=0)
    hr_ds.resize(curr_idx + add_count, axis=0)
    lr_ds[curr_idx:] = np.array(lr_list)
    hr_ds[curr_idx:] = np.array(hr_list)

def eval(args):
    h5_file = h5py.File(args.output_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, '*')))
    for i, image_path in enumerate(tqdm(image_paths, desc="Eval Data")):
        hr_pil = pil_image.open(image_path).convert('RGB')
        hr_y = rgb_to_ycbcr(hr_pil)[..., 0]
        
        w = (hr_y.shape[1] // args.scale) * args.scale
        h = (hr_y.shape[0] // args.scale) * args.scale
        hr_y = hr_y[:h, :w]

        hr_pil_y = pil_image.fromarray(hr_y.astype(np.uint8))
        lr_pil_y = hr_pil_y.resize((w // args.scale, h // args.scale), resample=pil_image.BICUBIC)

        lr_group.create_dataset(str(i), data=np.array(lr_pil_y).astype(np.float32))
        hr_group.create_dataset(str(i), data=np.array(hr_pil_y).astype(np.float32))

    h5_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--with-aug', action='store_true') # ميزة اختيارية للتدريب القوي
    parser.add_argument('--eval', action='store_true') # استخدم هذا العلم لإنشاء ملف التقييم
    args = parser.parse_args()

    if args.eval:
        eval(args)
    else:
        train(args)
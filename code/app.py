import glob
import multiprocessing
import os
import shutil
import subprocess
from zipfile import ZipFile

import imageio
import numpy as np
import tifffile
from flask import Flask, render_template, request
from google_drive_downloader import GoogleDriveDownloader as gdd
from validate_email import validate_email

app = Flask(__name__)


def convert_to_png(in_img):
    if in_img.endswith('.tiff') or in_img.endswith('.tif'):
        dat = tifffile.imread(in_img).astype(np.float)

    if len(dat.shape) != 2:
        # guess the data spec
        if dat.shape[-1] == 3:
            # Convert rgb to grey
            dat = 0.2989 * dat[:, :, 0] + 0.5870 * dat[:, :, 1] + 0.1140 * dat[
                                                                           :, :,
                                                                           2]
        elif dat.shape[-1] == 1:
            dat = dat[:, :, 0]
        else:
            raise RuntimeError("Unknown data structure: ", dat.shape, dat)
    assert dat.shape[0] == dat.shape[1] == 1024
    dat = dat.astype(np.uint8)
    # replace the file extension to png
    out_img = '.'.join(in_img.split('.')[:-1] + ['png'])
    imageio.imwrite(out_img, dat)
    return out_img


def collect_images_and_copy(train_dir, test_dir, out_dir):
    def collect_images(folder):
        images = []
        for valid_ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
            images.extend(
                glob.glob(f"{folder}/*.{valid_ext}"))
        return sorted(images)

    def move_or_copy_overwrite(src_f, dst_folder, move=True):
        dst_file = os.path.join(dst_folder, os.path.basename(src_f))
        if move:
            shutil.move(src_f, dst_file)
        else:
            shutil.copy(src_f, dst_file)

    train_signal_images = collect_images(f"{train_dir}/**/Signal/")
    """
    for valid_ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
        train_signal_images.extend(glob.glob(f"{train_dir}/**/Signal/*.{
        valid_ext}"))
    train_signal_images = sorted(train_signal_images)
    """
    train_target_images = [f.replace('Signal', 'Target') for f in
                           train_signal_images]
    train_A = os.path.join(out_dir, 'A/train')
    train_B = os.path.join(out_dir, 'B/train')
    os.makedirs(train_A, exist_ok=True)
    os.makedirs(train_B, exist_ok=True)
    for fa, fb in zip(train_signal_images, train_target_images):
        # Temperary!!
        fb = fb.replace('WHITE', 'F1')
        assert os.path.isfile(fb)
        # convert image to png format
        fa = convert_to_png(fa)
        fb = convert_to_png(fb)
        move_or_copy_overwrite(fa, train_A)
        move_or_copy_overwrite(fb, train_B)

    # prepare the test folder
    test_signal_images = collect_images(f"{test_dir}/**/Signal/")
    """
    test_signal_images = sorted(
        glob.glob(f"{test_dir}/**/Signal/*.*"))
    """
    test_A = os.path.join(out_dir, 'A/test')
    test_B = os.path.join(out_dir, 'B/test')
    os.makedirs(test_A, exist_ok=True)
    os.makedirs(test_B, exist_ok=True)
    for fa in test_signal_images:
        fa = convert_to_png(fa)
        move_or_copy_overwrite(fa, test_A, move=False)
        # a fake target image
        move_or_copy_overwrite(fa, test_B)


def combine_A_and_B(out_dir):
    bash_cmd = f"python ./scripts/combine_A_and_B.py --fold_A={out_dir}/A " \
               f"--fold_B={out_dir}/B --fold_AB={out_dir}/AB"
    ret_code = subprocess.run(bash_cmd.split())
    print(f"Combine A and B script finished with return code ({ret_code})")
    return ret_code


def train_marker(out_dir, expr_name):
    bash_cmd = f"python ./train.py --dataroot={out_dir}/AB " \
               f"--name {expr_name} " \
               f"--gpu_ids 1,2,3,4,5,7 --model=pix2pix --input_nc=1 " \
               f"--output_nc=1 --dataset_mode aligned --batch_size 24 " \
               f"--load_size=1024 --crop_size 1024"
    ret_code = subprocess.run(bash_cmd.split())
    print(f"Training code finished with return code ({ret_code})")
    return ret_code


def test_marker(out_dir, results_dir, expr_name):
    bash_cmd = f"python test.py --dataroot {out_dir}/AB " \
               f"--name {expr_name} --gpu_ids 7 --model pix2pix --input_nc 1 " \
               f"--output_nc 1 --dataset_mode aligned --load_size 1024 " \
               f"--crop_size 1024 --results_dir {results_dir}"
    ret_code = subprocess.run(bash_cmd.split())
    print(f"Testing code finished with return code ({ret_code})")
    return ret_code


def download_uri(file_id, email):
    print(f'Downloading {file_id}')
    zip_dest = f'third_party_data/{file_id}.zip'
    # unzip to file_id folder
    unzip_dest = f'third_party_data/{file_id}'
    if not os.path.isfile(zip_dest):
        gdd.download_file_from_google_drive(
            file_id=file_id,
            dest_path=zip_dest,
            unzip=False,
            showsize=True,
            overwrite=False)
        with ZipFile(zip_dest, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(unzip_dest)
    # extract the root folder name
    unzip_dest = glob.glob(f'{unzip_dest}/*/')[0]
    # move images to a separate folder namely A and B
    train_dir = os.path.join(unzip_dest, 'train')
    test_dir = os.path.join(unzip_dest, 'test')
    out_dir = os.path.join(unzip_dest, 'processed')
    results_dir = os.path.join(unzip_dest, 'results')

    print('Building training and testing folders')
    collect_images_and_copy(train_dir, test_dir, out_dir)
    print('Combining source and target data')
    combine_A_and_B(out_dir)
    # train for this marker
    print('Launching the training loop')
    expr_name = f"{file_id}"
    train_marker(out_dir, expr_name)
    # generate the prediction result
    print('Generating the prediction result')
    test_marker(out_dir, results_dir, expr_name)

    # Send email to client


@app.route('/start_training', methods=['POST'])
def start_training():
    result = request.form
    email = result['email']
    goog_uri = result['file_link']
    parts = goog_uri.split('/')
    try:
        idx_d = parts.index('d')
        file_id = parts[idx_d + 1]
        is_valid_uri = True
    except:
        is_valid_uri = False
        file_id = None
    is_valid_email = validate_email(email_address=email, check_regex=True,
                                    check_mx=False)
    if is_valid_email and is_valid_uri:
        # start a new process to download this file
        download_thread = multiprocessing.Process(
            target=lambda: download_uri(file_id, email),
            name=f'Download for {email}')
        download_thread.start()
    return render_template('success_page.html', result=result,
                           valid_email=is_valid_email, valid_uri=is_valid_uri)


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8990)
    #download_uri("1FBr8UjM6bMwidgmg_qyyvklCqmdK5Zkt", "xqliu@ucdavis.edu")

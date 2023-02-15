import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_path = './dataset'

image = Image.open(os.path.join(data_path, 'train-volume.tif'))
label = Image.open(os.path.join(data_path, 'train-labels.tif'))

# 이미지 사이즈
ny, nx  = label.size
nframe = label.n_frames

# train/test/val 폴더 생성하기
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(data_path, 'train')
dir_save_val = os.path.join(data_path, 'val')
dir_save_test = os.path.join(data_path, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 전체 이미지 30개를 섞어준다.
index_frame = np.arange(nframe)
np.random.shuffle(index_frame)

# train 이미지를 npy 파일로 저장하기
offset_nframe = 0

for i in range(nframe_train):
    label.seek(index_frame[i + offset_nframe])
    image.seek(index_frame[i + offset_nframe])

    label_ = np.asarray(label)
    image_ = np.asarray(image)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' %i), label_)
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' %i), image_)

# val 이미지를 npy 파일로 저장하기
offset_nframe += nframe_train

for i in range(nframe_val):
    label.seek(index_frame[i + offset_nframe])
    image.seek(index_frame[i + offset_nframe])

    label_ = np.asarray(label)
    image_ = np.asarray(image)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), image_)

# test 이미지를 npy 파일로 저장하기
offset_nframe += nframe_val

for i in range(nframe_test):
    label.seek(index_frame[i + offset_nframe])
    image.seek(index_frame[i + offset_nframe])

    label_ = np.asarray(label)
    image_ = np.asarray(image)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), image_)

    # 이미지 시각화
    plt.subplot(122)
    plt.imshow(label_, cmap='Blues')
    plt.title('label')

    plt.subplot(121)
    plt.imshow(image_, cmap='Blues')
    plt.title('image')

    plt.show()

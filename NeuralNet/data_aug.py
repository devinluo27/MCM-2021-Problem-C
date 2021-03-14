from PIL import ImageEnhance
from PIL import Image 
import os


root = os.getcwd() + '/'
dir_positive_ori = root + '/positive/'

dir_positive = root + '/positive_aug/'
if not os.path.exists(dir_positive):
    os.mkdir(dir_positive)
imgs = os.listdir(dir_positive_ori)
file_jpg = root + 'id_label_jpg.csv'
print(file_jpg)
f = open(file_jpg, 'a')
# f = open('a.csv', 'a')

for i in imgs:
    # original images
    raw_image = Image.open(dir_positive_ori + i)

    #旋转90°倍数
    rotate_90 = raw_image.rotate(90)
    rotate_180 = raw_image.rotate(180)
    rotate_270 = raw_image.rotate(270)

    #旋转结合翻转
    flip_vertical_raw = raw_image.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_90 = rotate_90.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_180 = rotate_180.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_270 = rotate_270.transpose(Image.FLIP_TOP_BOTTOM)
    i = i[:-4]

    #存储
    flip_vertical_raw.save(dir_positive + i + "_flip_vertical_raw.jpg")
    f.write(i + "_flip_vertical_raw.jpg, 1\n")
    flip_vertical_90.save(dir_positive + i + "_flip_vertical_90.jpg")
    f.write(i + "_flip_vertical_90.jpg, 1\n")
    flip_vertical_180.save(dir_positive + i + "_flip_vertical_180.jpg")
    f.write(i + "_flip_vertical_180.jpg, 1\n")
    flip_vertical_270.save(dir_positive + i + "_flip_vertical_270.jpg")
    f.write(i + "_flip_vertical_270.jpg, 1\n")

    raw_image.save(dir_positive +  i + '.jpg')

    rotate_90.save(dir_positive + i + "_rotate_90.jpg")
    f.write(i + "_rotate_90.jpg, 1\n")
    rotate_180.save(dir_positive + i + "_rotate_180.jpg")
    f.write(i + "_rotate_180.jpg, 1\n")
    rotate_270.save(dir_positive + i + "_rotate_270.jpg")
    f.write(i + "_rotate_270.jpg, 1\n")

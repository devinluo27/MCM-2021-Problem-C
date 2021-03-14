# create some enhanced imgs for paper writing
from PIL import ImageEnhance
from PIL import Image 
import os
import matplotlib.pyplot as plt


root = os.getcwd() + '/'
dir_positive_ori = root + '/positive/'

dir_positive = root + '/positive_aug_1/'
if not os.path.exists(dir_positive):
    os.mkdir(dir_positive)
# imgs = os.listdir(dir_positive_ori)
imgs = []
imgs.append(dir_positive_ori + 'ATT3067_IMG_20200926_091543.jpg')



for i in imgs:
    # original images
    raw_image = Image.open(i)
    # raw_image = raw_image.crop
    im_color = ImageEnhance.Color(raw_image).enhance(2.0)
    im_bri = ImageEnhance.Brightness(raw_image).enhance(2.0)
    im_con = ImageEnhance.Contrast(raw_image).enhance(1.8)
    im_sha = ImageEnhance.Sharpness(raw_image).enhance(2.0)

     #旋转90°倍数
    rotate_90 = raw_image.rotate(90)
    rotate_180 = raw_image.rotate(180)
    rotate_270 = raw_image.rotate(270)

    #旋转结合翻转
    flip_vertical_raw = raw_image.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_90 = rotate_90.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_180 = rotate_180.transpose(Image.FLIP_TOP_BOTTOM)
    flip_vertical_270 = rotate_270.transpose(Image.FLIP_TOP_BOTTOM)

    # im_bri.show()
    # im_color.show()

    plt.figure("Image") # 图像窗口名称
    # fig.tight_layout() # 调整整体空白
    plt.subplots_adjust(wspace =0.1, hspace =0.1)#调整子图间距


    plt.subplot(1,4,1)
    plt.imshow(raw_image)
    plt.title("Original", fontdict={'weight':'normal','size': 9}, y=-0.25)
    plt.axis('off') # 关掉坐标轴为 off


    plt.subplot(1,4,2)
    plt.imshow(rotate_90)
    plt.title("Rotate 90°", fontdict={'weight':'normal','size': 9}, y=-0.25)
    plt.axis('off') # 关掉坐标轴为 off

    plt.subplot(1,4,3)
    plt.imshow(flip_vertical_raw)
    plt.title("Flip Vertically", fontdict={'weight':'normal','size': 9}, y=-0.25)
    plt.axis('off') # 关掉坐标轴为 off

    plt.subplot(1,4,4)
    plt.imshow(flip_vertical_90)
    plt.title("Flip Vertical 90°", fontdict={'weight':'normal','size': 9}, y=-0.25)
    plt.axis('off') # 关掉坐标轴为 off


    plt.show()


    # flip_vertical_raw.save(dir_positive + i + "_flip_vertical_raw.jpg")



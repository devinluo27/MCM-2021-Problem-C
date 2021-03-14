from PIL import Image
import os

root = os.getcwd()
train_dir = root+'/train/'
files = os.listdir(train_dir)

# Change images to .jpg and remove the old .png images
print(train_dir + files[1][:-3])
count = 0
for file in files:
    count += 1
    if file[-3:] == 'png':
        im = Image.open(train_dir + file)
        rgb_im = im.convert('RGB')
        rgb_im.save(train_dir + file[:-3]+'jpg')
        os.remove(train_dir + file)

print(count)


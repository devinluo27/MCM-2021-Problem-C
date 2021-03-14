# modify id_label.csv

import csv
import os
import shutil

root = os.getcwd()
dir_positive = os.getcwd() +'/positive/'
if not os.path.exists(dir_positive):
    os.mkdir(dir_positive)
file_id_label_jpg = 'id_label_jpg.csv'
f_id_label_jpg = open(file_id_label_jpg)
train_dir = root + '/train/'
data_files = os.listdir(train_dir)

# list of all filenames in csv
data_lines = []
csv_data = csv.reader(f_id_label_jpg)
for one_line in csv_data:
    data_lines.append(one_line)

count = 0
for i in data_lines:
    if int(i[1]) == 1:
        print(i[0])
        shutil.copyfile(train_dir+i[0], dir_positive+i[0])
        count += 1

print(count)    

        


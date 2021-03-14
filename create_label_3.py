# modify id_label.csv
# create  datatxt without data_aug

import csv
import os

root = os.getcwd()
file_id_label_ori = 'id_label.csv'
file_id_label_ori_jpg = 'id_label_jpg.csv'
f_id_label_ori_jpg = open(file_id_label_ori_jpg, 'w')


file_id_label_mod = 'id_label_modified.csv'
f_id_label_ori = open(file_id_label_ori)
f_id_label_mod = open(file_id_label_mod, 'w')
train_dir = root + '/train/'
data_files = os.listdir(train_dir)

# list of all filenames in csv
data_lines = []
csv_data = csv.reader(f_id_label_ori)
for one_line in csv_data:
    data_lines.append(one_line)

print(data_lines[0])

# write png -> jpg
count = 0
for i in data_lines:
    if i[0][-3:] == 'png':
        i[0] = i[0][:-3] + 'jpg'
    if i[0] in data_files:
        count += 1
        f_id_label_ori_jpg.write('%s, %s\n'%(i[0], i[1].strip()))

print(count)

# count = 0
# for i in data_lines:
#     if i[0] in data_files:
#         count += 1
#         f_id_label_mod.write('%s, %s\n'%(i[0], i[1].strip()))

# print(count)

f_id_label_ori.close()
f_id_label_mod.close()
f_id_label_ori_jpg.close()

        


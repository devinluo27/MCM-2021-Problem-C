# delete useless files
import os
import csv

root = os.getcwd()
train_dir = root + '/train/'
data_files = os.listdir(train_dir)

s = []
invalid_cnt = 0
for file in data_files: #遍历文件夹
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
          if '.jpg' not in file and '.png' not in file:
              invalid_cnt += 1
              print(train_dir + file)
              os.remove(train_dir + file)

print('# of invalid files:', invalid_cnt) #打印结果

# print(files) 
# target_dir = os.walk(root)
# for path, dir_list, file_list in target_dir:  
#     print(path)
#     print()
#     # for file_name in file_list:  
#     # print(os.path.join(path, file_name) )


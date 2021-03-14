import os
import csv

file_dataset = 'dataset_utf8'
file_id = 'global_id_utf8'
file_id_label = 'id_label.csv'

file_unprocessed='unprocessed.csv'
file_unverified='unverified.csv'


f_dataset = open(file_dataset)
f_id = open(file_id)
f_id_label = open(file_id_label, 'w')
f_unverified = open(file_unverified, 'w')
f_unprocessed = open(file_unprocessed, 'w')


line = f_dataset.readline()
line = line.split(',')
data_lines = []
id_lines = []

csv_data = csv.reader(f_dataset)
for one_line in csv_data:
    data_lines.append(one_line)

print(data_lines[0])

csv_id = csv.reader(f_id)
for one_line in csv_id:
    id_lines.append(one_line)

print(id_lines[0])

f_id_label.write('%s, %s, %s, %s\n'%('file_name', 'positive', 'Latitude', 'Longitude'))
count = 0
for id in id_lines:
    for data in data_lines:
        if data[0] == id[1]:
            if 'egati' in data[3]:
                label = 0
                f_id_label.write('%s, %s, %s, %s\n'%(id[0], label, data[6], data[7]))
            if 'ositive' in data[3]:
                label = 1
                f_id_label.write('%s, %s, %s, %s\n'%(id[0], label, data[6], data[7]))
                count +=1

            break
        
print(count)
f_dataset.close()
f_id.close()
f_id_label.close()
f_unverified.close()
f_unprocessed.close()
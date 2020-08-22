import os
import argparse
import copy
import json

parser=argparse.ArgumentParser()
parser.add_argument('-f','--folder_name',type=int,required=False,default=10,help='Number of epochs to run the model')
args=parser.parse_args()

folder_name= ['train', 'test']

dir_numbers = list(range(0,10))


train_list = []
test_list = []

for fn in folder_name:

	total_path = '/users/mohita/nfs1_mohita/ufonia/speech_rec_pytorch/data/'+ fn + '/'

	# for n in dir_numbers:
	# 	current_folder = total_path + str(n) + '/'
		
	# 	if fn == 'train':
	# 		for file in os.listdir(current_folder):
	# 			train_list.append(file)
	# 	else:
	# 		for file in os.listdir(current_folder):
	# 			test_list.append(file)
	for file in os.listdir(total_path):
		if fn == 'train':
			train_list.append(file)
		else:
			test_list.append(file)

partition = {}
partition['train'] = train_list
partition['test'] = test_list

labels = {}
for i in train_list:
	label_this = int(i.split("_")[0])
	labels[i] = label_this

for i in test_list:
	label_this = int(i.split("_")[0])
	labels[i] = label_this


with open('/users/mohita/nfs1_mohita/ufonia/speech_rec_pytorch/data/partition_aug.json', 'w') as fp:
    json.dump(partition, fp)

with open('/users/mohita/nfs1_mohita/ufonia/speech_rec_pytorch/data/labels_aug.json', 'w') as fp:
    json.dump(labels, fp)

# import pdb; pdb.set_trace()



import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,csv,pickle
from PIL import Image

def create_meta_csv(dataset_path, destination_path):

	# Change dataset path accordingly
	DATASET_PATH = os.path.abspath(dataset_path)
	
	if destination_path == None:
		destination_path = dataset_path

	# Change destination path accoridingly
	DEST_PATH = os.path.abspath(destination_path)
	dataset_name = DATASET_PATH.split(os.sep)[-1]
	dataset_file_name = dataset_name.split(' ')[0].lower()
	file_path = os.path.join(DEST_PATH,dataset_file_name+'.csv')
	
	print('File:-',dataset_file_name)

	classes = {}
	rev_classes = {}
	with open(file_path,'w') as f:	
		wr = csv.writer(f)
		i = 0
		ct = 0
		for subdir, dirs, files in os.walk(DATASET_PATH):
			subd = subdir.split(os.sep)[-1]		
			if subd == dataset_name:
				wr.writerow(['path','label'])
				continue

			for file in files:
				class_name = file[0]
				if class_name in rev_classes:
					pass
				else:
					
					classes[i] = class_name
					rev_classes[class_name] = i
					print(classes)
					i+=1
				wr.writerow([os.path.join(subdir, file),rev_classes[class_name]])
			
		f.close()

	with open(dataset_file_name+'_model_dict'+'.pkl', 'wb') as f:
		pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)	

	return True
					
def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
	if create_meta_csv(dataset_path, destination_path=destination_path):

		DATASET_PATH = os.path.abspath(dataset_path)
		dataset_name = DATASET_PATH.split(os.sep)[-1].split(' ')[0].lower()
		dframe = pd.read_csv(os.path.join(destination_path, dataset_name+'.csv'))

		if randomize == True or (split != None and randomize == None):
			# shuffle the dataframe here
			dframe = dframe.sample(frac=1).reset_index(drop=True)

		if split != None:
			train_set, test_set = train_test_split(dframe, split)
			return dframe, train_set, test_set

		return dframe

def train_test_split(dframe, split_ratio):
	"""Splits the dataframe into train and test subset dataframes.

	Args:
		split_ration (float): Divides dframe into two splits.

	Returns:
		train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
		test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
	"""
	# divide into train and test dataframes
	train_data = dframe.sample(frac=split_ratio).reset_index(drop=True)
	test_data = dframe.drop(train_data.index).reset_index(drop=True)
	return train_data, test_data


class ImageDataset(Dataset):
	"""Image Dataset that works with images

	This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
	Args:
		data (str): Dataframe with path and label of images.
		transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.

	Examples:
		>>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
		>>> train_dataset = dataset.ImageDataset(train_df)
		>>> test_dataset = dataset.ImageDataset(test_df, transform=...)
	"""

	def __init__(self, data, transform=None):
		self.data = data
		self.transform = transform
		self.classes = data['label'].unique()# get unique classes from data dataframe
		# print(self.classes) # Uncomment to print unique classes

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data.iloc[idx]['path']
		image = Image.open(img_path).convert('RGB')# load PIL image
		label = self.data.iloc[idx]['label'] # get label (derived from self.classes; type: int/long) of image
		# image.show()   # Uncomment to see the image
		if self.transform:
			image = self.transform(image)

		return image, label

if __name__ == '__main__':
	dataset_path = './NUS Hand Posture dataset-II'
	destination_path = './'

	df, tdf, tstdf = create_and_load_meta_csv_df(dataset_path, destination_path,randomize=True,split=0.2)


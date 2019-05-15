# import cv2,math,os,argparse,time
# import numpy as np
# from image_utils import *
import torch, torchvision, os, pickle, warnings, argparse
from torchvision import transforms
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

data_transforms = {
	'test': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load('./nus_model.pth', map_location=device)
model = model.to(device)

mapping = os.path.abspath('./nus_model_dict.pkl')

with open(mapping, 'rb') as f:
	gestures_mapping = pickle.load(f)

def process_image(img, transforms):
	image = Image.open(img)
	image_tensor = transforms(image)

	# Add an extra dimension to image tensor representing batch size
	image_tensor = image_tensor.unsqueeze_(0)
	return image_tensor

def get_gesture_name(image_path):

	# Process the image
	image = process_image(image_path, data_transforms['test'])

	# set model to evaluation mode
	model.eval()

	image = image.to(device)
	outputs = model(image)
	_, preds = torch.max(outputs, 1)

	# predicted habitat
	return gestures_mapping[preds.item()]

# Argument parsing
parser = argparse.ArgumentParser()

# Adding parser arguments
parser.add_argument('path' ,help="Image path")
args=parser.parse_args()

image_path = os.path.abspath(args.path)
gesture = get_gesture_name(image_path)

print("Recognized gesture is: ",gesture)

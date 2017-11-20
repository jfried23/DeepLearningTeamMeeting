import glob
import PIL
import numpy as np
from sklearn import model_selection

p='/Users/frjo6001/Documents/DeepLearningTeamMeeting/dogs_v_cats/data/train/'
all_files = glob.glob(p+'*')

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))

# First resize and convert images to numpy array
y   = np.empty((len(all_files)), np.int)
x   = np.empty( (len(all_files),224,224,3), np.uint8 )

for i,ff in enumerate(all_files):
	flg = 0 if 'cat' in ff.split('/')[-1] else 1

	y[i] = flg
	x[i] = np.array(PIL.Image.open(ff).resize((224,224)))


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=.2,stratify=y)
del x
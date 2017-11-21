import numpy as np 
import keras
from keras.utils.np_utils import to_categorical

d=np.load('./dset.npz')


#Setup model

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
vgg_norm = keras.layers.Lambda(lambda x: x[:,:,::-1]-vgg_mean, 
                               output_shape=lambda x: x)

input_imgs = keras.layers.Input((224,224,3))
x          = vgg_norm(input_imgs)

vgg16 = keras.applications.VGG16(include_top=False, 
                                 input_tensor=x,
                                 pooling='avg')

for l in vgg16.layers: l.trainable = False


out_test  = vgg16.predict(d['x_test'],verbose=1,batch_size=32)
out_train = vgg16.predict(d['x_train'],verbose=1,batch_size=32)


inp = keras.layers.Input((512,))
x = keras.layers.Dropout(0.5)(inp)
out = keras.layers.Dense(1,activation=keras.activations.sigmoid)(x)
model = keras.models.Model(inp,out)
model.compile(keras.optimizers.Adam(), keras.losses.binary_crossentropy,metrics=['accuracy'])
model.fit( x=out_train, y=np.atleast_2d(d['y_train']).T,
	       batch_size=32, epochs=30, 
	       validation_data=(out_test,np.atleast_2d(d['y_test']).T),
	       verbose=1
	       )
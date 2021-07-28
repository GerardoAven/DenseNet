# Librerías a utilizar para el ejemplo
import numpy as np  
from scipy import misc  
from PIL import Image  
import glob  
import matplotlib.pyplot as plt  
import scipy.misc  
from matplotlib.pyplot import imshow  
import cv2  
import seaborn as sn  
import pandas as pd  
import pickle  
from keras import layers  
from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout  
from keras.models import Sequential, Model, load_model  
from keras.preprocessing import image  
from keras.preprocessing.image import load_img  
from keras.preprocessing.image import img_to_array  
from keras.applications.imagenet_utils import decode_predictions  
from keras.utils import layer_utils, np_utils  
from keras.utils.data_utils import get_file  
from keras.applications.imagenet_utils import preprocess_input  
from keras.utils.vis_utils import model_to_dot  
from keras.utils import plot_model  
from keras.initializers import glorot_uniform  
from keras import losses  
import keras.backend as K  
from keras.callbacks import ModelCheckpoint  
from sklearn.metrics import confusion_matrix, classification_report  
import tensorflow as tf

# Librerías que se necesitan para usar la arquitectura DenseNet
from keras.applications import densenet  
from keras.applications import imagenet_utils as imut

# Función para la customización de DenseNet
def CustomDenseNet(blocks, include_top=True, input_tensor=None, weights=None, input_shape=None, pooling=None, classes=1000):

	# Determinar la forma correcta de la entrada
	input_shape = imut._obtain_input_shape(input_shape, default_size=224, min_size=32, data_format=K.image_data_format(), require_flatten=include_top, weights=weights)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

	x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
	x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
	x = Activation('relu', name='conv1/relu')(x)
	x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
	x = MaxPooling2D(3, strides=2, name='pool1')(x)

	x = densenet.dense_block(x, blocks[0], name='conv2')
	x = densenet.transition_block(x, 0.5, name='pool2')
	x = densenet.dense_block(x, blocks[1], name='conv3')
	x = densenet.transition_block(x, 0.5, name='pool3')
	x = densenet.dense_block(x, blocks[2], name='conv4')
	x = densenet.transition_block(x, 0.5, name='pool4')
	x = densenet.dense_block(x, blocks[3], name='conv5')

	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

	if include_top:
		x = GlobalAveragePooling2D(name='avg_pool')(x)
		x = Dense(classes, activation='softmax', name='fc1000')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling2D(name='avg_pool')(x)
		elif pooling == 'max':
			x = GlobalMaxPooling2D(name='max_pool')(x)

	if input_tensor is not None:
		inputs = imut.get_source_inputs(input_tensor)
	else:
		inputs = img_input

    # Creación del modelo.
	if blocks == [6, 12, 24, 16]:
		model = Model(inputs, x, name='densenet121')
	elif blocks == [6, 12, 32, 32]:
		model = Model(inputs, x, name='densenet169')
	elif blocks == [6, 12, 48, 32]:
		model = Model(inputs, x, name='densenet201')
	else:
		model = Model(inputs, x, name='densenet')

    # Cargar los pesos.
	if weights == 'imagenet':
		if include_top:
			if blocks == [6, 12, 24, 16]:
				weights_path = get_file('densenet121_weights_tf_dim_ordering_tf_kernels.h5', DENSENET121_WEIGHT_PATH, cache_subdir='models', file_hash='0962ca643bae20f9b6771cb844dca3b0')
			elif blocks == [6, 12, 32, 32]:
				weights_path = get_file('densenet169_weights_tf_dim_ordering_tf_kernels.h5', DENSENET169_WEIGHT_PATH, cache_subdir='models', file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
			elif blocks == [6, 12, 48, 32]:
				weights_path = get_file('densenet201_weights_tf_dim_ordering_tf_kernels.h5', DENSENET201_WEIGHT_PATH, cache_subdir='models', file_hash='7bb75edd58cb43163be7e0005fbe95ef')
		else:
			if blocks == [6, 12, 24, 16]:
				weights_path = get_file('densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET121_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
			elif blocks == [6, 12, 32, 32]:
				weights_path = get_file('densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET169_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='50662582284e4cf834ce40ab4dfa58c6')
			elif blocks == [6, 12, 48, 32]:
				weights_path = get_file('densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET201_WEIGHT_PATH_NO_TOP, cache_subdir='models', file_hash='1c2de60ee40562448dbac34a0737e798')
			model.load_weights(weights_path)
	elif weights is not None:
		model.load_weights(weights)

	return model

# Creación de la red DenseNet
def create_densenet():  
	base_model = CustomDenseNet([6, 12, 16, 8], include_top=False, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=100)
	x = base_model.output

	x = GlobalAveragePooling2D(name='avg_pool')(x)
	x = Dense(500)(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(100, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	return model

# Creación de la red con las funciones creadas  
custom_dense_model = create_densenet()  
custom_dense_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])

#Resumen del modelo creado
custom_dense_model.summary()

#Entrenamiento del modelo
cdense = custom_dense_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)

# Gráficas de las métricas del modelo para su entrenamiento y validación.
plt.figure(0)  
plt.plot(cdense.history['acc'],'r')  
plt.plot(cdense.history['val_acc'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(cdense.history['loss'],'r')  
plt.plot(cdense.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])

plt.show()

# Matriz de confusión
cdense_pred = custom_dense_model.predict(x_test, batch_size=32, verbose=1)  
cdense_predicted = np.argmax(cdense_pred, axis=1)

cdense_cm = confusion_matrix(np.argmax(y_test, axis=1), cdense_predicted)

# Visualización de la matriz de confusión
cdense_df_cm = pd.DataFrame(cdense_cm, range(100), range(100))  
plt.figure(figsize = (20,14))  
sn.set(font_scale=1.4)
sn.heatmap(cdense_df_cm, annot=True, annot_kws={"size": 12})
plt.show()

# Métricas
cdense_report = classification_report(np.argmax(y_test, axis=1), cdense_predicted)  
print(cdense_report)

#Curva ROC
from sklearn.datasets import make_classification  
from sklearn.preprocessing import label_binarize  
from scipy import interp  
from itertools import cycle

n_classes = 100

from sklearn.metrics import roc_curve, auc

lw = 2

# Cálculo de la cruva ROC y el área ROC
fpr = dict()  
tpr = dict()  
roc_auc = dict()  
for i in range(n_classes):  
	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], cdense_pred[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), cdense_pred.ravel())  
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)  
for i in range(n_classes):  
	mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr  
tpr["macro"] = mean_tpr  
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(1)  
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(n_classes-97), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()


plt.figure(2)  
plt.xlim(0, 0.2)  
plt.ylim(0.8, 1)  
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])  
for i, color in zip(range(3), colors):  
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Some extension of Receiver operating characteristic to multi-class')  
plt.legend(loc="lower right")  
plt.show()  

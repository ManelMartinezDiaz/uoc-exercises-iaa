# Cargar la base de datos MNIST y librerias Keras:
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

######################################
#Afegir Exercici2a
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=200, resize=1, color = False, funneled=True)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print("h:",h)
print("w:",w)

# Establecer la semilla del generador de numeros aleatorios para garantizar reproducibilidad de los resultados 
seed = 7
numpy.random.seed(seed)

# Acceder a las imagenes de entrenamiento y test
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Dimensionar las imagenes en vectores 
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_train = X_train.reshape(X_train.shape[0],1,125,94)
X_test = X_test.reshape(X_test.shape[0],1,125,94)

# Los datos de entrada (pixeles) son enteros,
# convertirlos a reales para trabajar con ellos
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizar las imagenes de escala de grises 0-255 (valores entre 0-1):
X_train = X_train / 255
X_test = X_test / 255

# Codificar las etiquetas de clase en formato de vectores categoricos con diez posiciones: 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Funcion en la que se define la arquitectura del modelo:
def baseline_model():
# Capa de entrada:
	model = Sequential() 
# Primera capa (Convolucional):
	model.add(Conv2D(32, (5, 5), input_shape=(1, 125,94), activation='relu'))
# Tercera capa (agrupamiento):
	model.add(MaxPooling2D(pool_size=(2, 2))) 
# Cuarta capa (regularizacion)
    #model.add(Dropout(0.2))
	model.add(Dropout(0.2))
# Quinta capa (redimensionamiento): 
	model.add(Flatten())
# Sexta capa (completamente conectada)
	model.add(Dense(128, activation = 'relu'))
   #model.add(Dense(128, activation='relu'))
    
# Capa de salida (softmax):
	model.add(Dense(num_classes, activation='softmax')) 
# Compilar el modelo y especificar metodo y metrica de optimizacion:
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
	return model

# Llamada al modelo: 
model = baseline_model()

# Ajustar el modelo utilizando los datos de entrenamiento y validacion con los de test:
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=500, verbose=2)

# Evaluacion del modelo utilizando los datos de test:
scores = model.evaluate(X_test, y_test, verbose=0)
print("Exactitud del modelo: %.2f%%" % (100*scores[1]))


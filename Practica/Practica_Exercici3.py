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

#Agefir Exercici3a
from keras.models import Model
from keras.layers import Input
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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
# Definir el modelo de la red, con dos capas ocultas
    #Capa d ’ entrada :
    model = Sequential()
    # Primera capa ( convolucional ) :
    #model.add(Conv2D(32,(5, 5), #Modificar
        #activation='relu',
            #input_shape=(1, 28, 28)))
    model.add(Conv2D(32,(5, 5),                      
                     activation='relu',
                     input_shape=(1, 125, 94)))
    #Tercera capa ( agrupament ):
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Quarta capa ( r e gul a r i t z a c io ):
    model.add(Dropout(0.2))
    # Cinquena capa ( redimensionament ) :
    model.add(Flatten())
    # Si sena capa ( completament connectada )
    model.add(Dense(128, activation = 'relu'))
    # Capa de sor t ida :
    model.add(Dense(num_classes, activation= 'softmax'))
    
    # Compilar y entrenar
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model

#Exercici 3a Funció per arquitectura amb funció 'Model'
def nou_model():
# Capa de entrada:
	model = Sequential() 
# Primera capa (Convolucional):
	model.add(Conv2D(32, (5, 5), input_shape=(1, 125,94), activation='relu'))
# Tercera capa (agrupamiento):
	model.add(MaxPooling2D(pool_size=(2, 2))) 
# Cuarta capa (regularizacion)
	model.add(Dropout(0.2))
# Quinta capa (redimensionamiento): 
	model.add(Flatten())
# Sexta capa (completamente conectada)
	model.add(Dense(128, activation = 'relu', name='f'))
   #model.add(Dense(128, activation='relu'))
# Capa de salida (softmax):
	model.add(Dense(num_classes, activation='softmax')) 
# Compilar el modelo y especificar metodo y metrica de optimizacion:
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


       
# Llamada al modelo: 
#model = baseline_model()
model = nou_model()
model1 = Model(inputs=model.input, outputs=model.get_layer('f').output)  

#Obtenim el conjunt de dades on aplicar el classificador
X_train_cnn = model1.predict(X_train)
#print("Shape_X_train_cnn:", X_train_cnn.shape)
#X_test_cnn = model1.predict(X_test)
#print("Shape_X_test_cnn:", X_test_cnn.shape)
#feat_val = model1.predict(y_test)
#print(feat_val.shape)

# #############################################################################
# Aplicar els classificadores sobre el nou conjunt X-train_cnn
print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

clf.fit(X_train_cnn,numpy.argmax(y_train,axis=1))
#print("Prediction Score-Training attributes:", clf.score(X_train_cnn,numpy.argmax(y_train,axis=1)))

# #############################################################################
#Exercici 1a
print("Cross_Val_Score:",cross_val_score(clf,X_train_cnn,numpy.argmax(y_train,axis=1), cv=10).mean())


# Avaluacio del model u t i l i t z ant l e s dades de prova :
#scores = clf.score(X_test_cnn, y_test)
#print ( "Exactitud del model: ", scores*100)

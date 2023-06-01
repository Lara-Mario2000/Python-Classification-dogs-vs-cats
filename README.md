# Python: clasificación de fotos de perros y gatos
## Introducción del proyecto
El objetivo del proyecto es desarrollar mi propia versión de una red neuronal convolucional profunda para clasificar fotografías de perros y gatos, se desarrollo con la ayuda del siguiente enlace: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

Líder:	Mario Eduardo Lara Loredo
Fechas:	31/05/2023

# Detalles del proyecto
Información del conjunto de datos
El conjunto de datos que se descargó de Kaggle contiene 2 carpetas (Train y Test), ambas carpetas subdividas en otras 2 carpetas (Cats y Dogs). Que quedan de la siguiente manera:
Dataset
|-------	Train
|	|-------	Cats
|	└-------	Dogs
└-------	Test
|-------	Cats
└-------	Dogs
La carpeta “Train” contiene 25000 imágenes de gatos y perros, 12500 para gatos y 12500 para perros, cada imagen etiqueta por nombre de la imagen en el siguiente formato: 
cat.n.jpg,
dog.n.jpg
donde n es el número.
Modelo CNN
La tarea principal de la clasificación de imágenes es la aceptación de la imagen de entrada y la siguiente definición de su clase. Un humano puede determinar fácilmente que la criatura en la imagen es un elefante. Pero la computadora ve las imágenes de manera muy diferente:
 
En lugar de la imagen, la computadora ve una matriz de píxeles. Por ejemplo, si el tamaño de la imagen es 300 x 300. En este caso, el tamaño de la matriz será 300x300x3. Donde 300 es ancho, siguiente 300 es alto y 3 son valores de canal RGB. Al equipo se le asigna un valor de 0 a 255 para cada uno de estos números. Este valor describe la intensidad del píxel en cada punto.
Para resolver este problema el ordenador busca las características del nivel base. En la comprensión humana, tales características son, por ejemplo, el tronco o las orejas grandes. Para la computadora, estas características son límites o curvaturas. Luego, a través de los grupos de capas convolucionales, la computadora construye conceptos más abstractos.
Las redes neuronales convolucionales funcionan ingiriendo y procesando grandes cantidades de datos en formato cuadrícula y extrayendo después importantes características detalladas para su clasificación y detección. Las CNN suelen componerse de tres tipos de capas: una capa convolucional, una capa de agrupación y una capa totalmente conectada. Cada capa tiene un propósito distinto, realiza una tarea sobre los datos ingeridos y aprende cantidades crecientes de complejidad. Esta arquitectura permite que las CNN aprendan automáticamente patrones visuales complejos y representaciones jerárquicas de las imágenes.
 
GPU para aprendizaje profundo
Para cualquier red neuronal, la fase de entrenamiento del modelo de aprendizaje profundo es la tarea que consume más recursos. Durante el entrenamiento, una red neuronal recibe entradas, que luego se procesan en capas ocultas utilizando pesos que se ajustan durante el entrenamiento y el modelo luego escupe una predicción. Los pesos se ajustan para encontrar patrones con el fin de hacer mejores predicciones.
Ambas operaciones son esencialmente multiplicaciones de matrices. Una simple multiplicación de matrices puede ser representada por la imagen de abajo
 
Suponiendo que una red neuronal tiene alrededor de 10, 100 o incluso 100,000 parámetros. Una computadora aún podría manejar esto en cuestión de minutos, o incluso horas como máximo. Sin embargo, la realidad es que las redes neuronales tienen mas de 10 mil millones de parámetros, lo que llevaría años entrenar este tipo de modelos empleando el enfoque tradicional. Los modelos de aprendizaje profundo se pueden entrenar más rápido simplemente ejecutando todas las operaciones al mismo tiempo en lugar de una tras otra. Se puede lograr esto usando una GPU para entrenar el modelo. 
Las GPU están optimizadas para entrenar modelos de inteligencia artificial y aprendizaje profundo, ya que pueden procesar múltiples cálculos simultáneamente. Tienen una gran cantidad de núcleos, lo que permite un mejor cálculo de múltiples procesos paralelos. Además, los cálculos en el aprendizaje profundo necesitan manejar grandes cantidades de datos, lo que hace que el ancho de banda de memoria de una GPU sea el más adecuado.
Requisitos
Las especificaciones técnicas en que se realizo el proyecto son las siguientes:
SO	Windows 11
Procesador	Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
RAM instalada	16.0 GB (15.9 GB utilizable)
GPU	NVIDIA GeForce GTX 1660 Ti

Adicionalmente las versiones de los softwares utilizados son las siguientes:
•	python=3.9
•	cudatoolkit=11.2 
•	cudnn=8.1.0
•	tensorflow<2.11
Para facilitar la instalación de estos componentes se empleó conda=23.3.1
Instalación del ambiente:
1.	Descargar miniconda del siguiente enlace: Miniconda — conda documentation y seguir los pasos del ejecutable.
2.	Una vez termine la instalación, es necesario crear un nuevo ambiente que contenga Python 3.9.
a.	Para realizarlo debe ingresar a la consola de conda
 
b.	Ejecutar la siguiente línea:
conda create --name snakes python=3.9
c.	Esto creara un ambiente llamado “snakes” el cual será el que se utilizara para el proyecto. Para activarlo se utiliza la siguiente línea:
conda activate snakes
3.	Con la siguiente línea, se instalan las herramientas que permiten a tensorflow trabajar con la GPU:
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
4.	Finalmente se instala tensorflow 2.10
python -m pip install "tensorflow<2.11"
Descripción del Código
Preparación del dataset
Importante: Los siguientes pasos son opcionales. “Preprocesamiento de fotos” sirve para tener todas las fotos en un mismo tamaño desde antes que se cree el modelo, sin embargo durante la creación del modelo también se hace un escalado, para estandarizar las imágenes. Mientras que “Preprocesamiento de directories” es en caso en que el directorio de “train” y “test” no este separado en subcarpetas como se solicita. En cualquier caso, si se cumple con estos requisitos se puede saltar a la siguiente parte.
Preprocesamiento de fotos
Mirando algunas fotos aleatorias en el directorio, se puede ver que las fotos son en color y tienen diferentes formas y tamaños. Esto sugiere que las fotos tendrán que ser remodeladas antes de modelar para que todas las imágenes tengan la misma forma. Esta es a menudo una pequeña imagen cuadrada.
Preprocesamiento de directorios
Alternativamente, se pueden cargar las imágenes progresivamente usando la clase Keras ImageDataGenerator y la API flow_from_directory(). Esta API prefiere que los datos se dividan en directorios separados train/ y test/, y debajo de cada directorio tener un subdirectorio para cada clase, por ejemplo, un tren/perro/ y un tren/gato/ subdirectorios y lo mismo para la prueba. Las imágenes se organizan en los subdirectorios.
Modelo de aprendizaje por transferencia
El aprendizaje por transferencia (transfer learning en inglés) es una técnica utilizada en el campo del aprendizaje automático y la inteligencia artificial, que consiste en aprovechar y transferir el conocimiento adquirido por un modelo previamente entrenado en una tarea hacia una nueva tarea relacionada.
En lugar de entrenar un modelo desde cero para una tarea específica, se toma un modelo previamente entrenado en una tarea similar y se ajusta o se utiliza como punto de partida para la nueva tarea. Esto permite aprovechar el conocimiento y las representaciones aprendidas por el modelo en la tarea previa, que a menudo incluyen características visuales y conceptuales útiles, y aplicarlos en la nueva tarea.
Keras proporciona una gama de modelos previamente entrenados que se pueden cargar y utilizar total o parcialmente a través de la API de aplicaciones Keras. Un modelo útil para el aprendizaje de transferencia es uno de los modelos VGG, como VGG-16 con 16 capas que en el momento en que se desarrolló, logró los mejores resultados en el desafío de clasificación de fotos de ImageNet.
El modelo se compone de dos partes principales, la parte extractora de características del modelo que se compone de bloques VGG, y la parte clasificadora del modelo que se compone de capas totalmente conectadas y la capa de salida. Podemos usar la parte de extracción de características del modelo y agregar una nueva parte clasificadora del modelo que se adapte al conjunto de datos de perros y gatos. Específicamente, podemos mantener los pesos de todas las capas convolucionales fijas durante el entrenamiento, y solo entrenar nuevas capas totalmente conectadas que aprenderán a interpretar las características extraídas del modelo y hacer una clasificación binaria.
Esto se puede lograr cargando el modelo VGG-16, eliminando las capas totalmente conectadas del extremo de salida del modelo, luego agregando las nuevas capas completamente conectadas para interpretar la salida del modelo y hacer una predicción. La parte clasificadora del modelo se puede eliminar automáticamente estableciendo el argumento "include_top" en "False", que también requiere que la forma de la entrada también se especifique para el modelo, en este caso (224, 224, 3). Esto significa que el modelo cargado termina en la última capa de agrupación máxima, después de lo cual podemos agregar manualmente una capa Acoplar y las nuevas capas clasificadoras. La función define_model() siguiente implementa esto y devuelve un nuevo modelo listo para el entrenamiento.
# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
Marco de pruebas
Se puede llamar para preparar un modelo según sea necesario, por ejemplo:
model = define_model()
A continuación, tenemos que preparar los datos, ya que el modelo también espera que las imágenes estén centradas. Es decir, tener los valores medios de píxeles de cada canal (rojo, verde y azul) calculados en el conjunto de datos de entrenamiento de ImageNet restados de la entrada. Keras proporciona una función para realizar esta preparación para fotos individuales a través de la función preprocess_input(). Sin embargo, podemos lograr el mismo efecto con ImageDataGenerator estableciendo el argumento "featurewise_center" en "True" y especificando manualmente los valores de píxeles medios que se usarán al centrar como valores medios del conjunto de datos de entrenamiento de ImageNet: [123.68, 116.779, 103.939].
A continuación, los iteradores deben estar preparados tanto para el tren como para los conjuntos de datos de prueba. Se usa la funcion flow_from_directory() en el generador de datos y crear un iterador para cada uno de los directorios train/ y test/. Se debe especificar que el problema es un problema de clasificación binaria a través del argumento "class_mode", y cargar las imágenes con el tamaño de 224×224píxeles a través del argumento "target_size". Fijaremos el tamaño del lote en 64. 
Se ajusta el modelo utilizando el iterador de tren (train_it) y usar el iterador de prueba (test_it) como conjunto de datos de validación durante el entrenamiento. Se debe especificar el número de pasos para el tren y los iteradores de prueba. Este es el número de lotes que comprenderán una época. Esto se puede especificar a través de la longitud de cada iterador, y será el número total de imágenes en los directorios de tren y prueba dividido por el tamaño del lote (64). El modelo será apto para 10 épocas, un pequeño número para comprobar si el modelo puede aprender el problema. Una vez ajustado, el modelo final se puede evaluar directamente en el conjunto de datos de prueba y se puede informar la precisión de la clasificación. 
A continuacion, se puede crear un gráfico del historial recopilado durante el entrenamiento almacenado en el directorio "historial" devuelto de la llamada a fit_generator(). El historial contiene la precisión y la pérdida del modelo en el conjunto de datos de prueba y entrenamiento al final de cada época. Los diagramas de líneas de estas medidas durante las épocas de entrenamiento proporcionan curvas de aprendizaje que podemos usar para tener una idea de si el modelo está sobreajustado, mal ajustado o tiene un buen ajuste. Finalmente para guardar el modelo final en un archivo H5 llamando a la función save() en el modelo y pasar el nombre de archivo elegido.
Diagnóstico del modelo
La función summarize_diagnostics() a continuación toma el directorio de historial y crea una sola figura con un gráfico de líneas de la pérdida y otro para la precisión. A continuación, la figura se guarda en un archivo con un nombre de archivo basado en el nombre del script. Esto es útil si deseamos evaluar muchas variaciones del modelo en diferentes archivos y crear gráficos de líneas automáticamente para cada uno.
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
Predicciones
Se puede usar el modelo guardado para hacer una predicción sobre nuevas imágenes. El modelo asume que las nuevas imágenes son en color y se han segmentado para que una imagen contenga al menos un perro o gato. Primero, podemos cargar la imagen y forzarla a un tamaño de 224×224 píxeles. A continuación, se puede cambiar el tamaño de la imagen cargada para tener una sola muestra en un conjunto de datos. Los valores de píxeles también deben estar centrados para que coincidan con la forma en que se prepararon los datos durante el entrenamiento del modelo. La función load_image() implementa esto y devolverá la imagen cargada lista para su clasificación. A continuación, podemos cargar el modelo como en la sección anterior y llamar a la función predict() para predecir el contenido de la imagen como un número entre "0" y "1" para "gato" y "perro" respectivamente.
Bibliografía
https://github.com/hosamelsafty/Cats-VS-Dogs
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

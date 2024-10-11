## Bibliografia

### Contextual
- [Reconocimiento usando low cam - UPC (2019)](obsidian://open?vault=ProcesamientoImg&file=Proyecto%2FBerru_NB_Reconocimiento_gestos_estaticos(2021).pdf)

	Modelo de clasificacion
	1. Construyeron el dataset, validado por [ENSENAS](https://ensenaperu.org/).
	2. Preprocesamiento: extraccion de fondo + HOG ->
	3. Vector de caractersticas ->
	4. SVM, KNN, MLP, RF

	Summary
	La idea, es asistir al usuario en el proceso de aprendizaje del lenguaje de senas. 

	 Dataset > 
		 3600 imágenes de los gestos estáticos del lenguaje de señas peruano (alfabeto) con aplicacion en un sistema de reconocimiento de gestos con cámaras de baja resolución
	 
	 Proceso
		![[Pasted image 20241009160452.png]]


	 Metodos
	 
	 1.  Background sustraccion  
	 2. Descriptor de características de Histograma de Gradientes Orientados (HOG)
	 3. 4 algoritmos de clasificación. 

	 Resultados
	 
	 Los resultados mostraron que utilizando el **Histograma de Gradientes Orientados**  +  el algoritmo de clasificación de **Support Vector Machine**, se obtuvo el mejor resultado con un accuracy de 89.46% y el sistema pudo recono
	 cer los gestos con **variaciones** de traslación, rotación y escala.

	 Sugieren realizar investigaciones con DL para compararlos con los resultados del proyecto
	 [[Berru_NB_Reconocimiento_gestos_estaticos(2021).pdf#page=28&selection=14,37,17,47|Berru_NB_Reconocimiento_gestos_estaticos(2021), page 28]] 

	 Fuentes por proceso:

- [Reconocimiento desde WebCam ML- UAC (2021)](https://repositorio.uandina.edu.pe/item/d6ae0bf9-0a8e-4cc3-a79f-5df4355141be)
	- No explorado
- [Tecnicas de Clusteting - Ricardo Palma (2020)](obsidian://open?vault=ProcesamientoImg&file=Proyecto%2FTECNICAS_CLUSTERING_ESPINOZA%20.pdf)

	No encuentro el DataSet
	Clustering, Redes neuronales, alfabeto dactilogico peruano.
	Parametros distintivos: 
	 1. la tabla de contingencia,
	 2. precisión 
	 3. recall 
	Regresión General  y Aprendizaje por Cuantización Vectorial
	3 redes neuronales distintas c/u: 
	1. una encargada de las señas
	2. otra para los sensores MPU 
	3. y una red conjugada que congrega los resultados de las redes previas, y la variable dependiente definida por el reconocimiento de los gestos estáticos 
	Resultados, 
	 1. Red Neuronal de Regresión Generalizada posee una gran capacidad en el reconocimiento de estos gestos, con una precisión y recall del 100% verificada en su tabla de contingencia para datos pre procesados, así como una resistencia al ruido del 1% para mantener esta efectividad; 
	 2. Red Neuronal de Aprendizaje por Cuantización Vectorial requiere de una mayor cantidad de sensores, así como la adición de una red distinta para poder ser más preciso en su reconocimiento, con valores de precisión y recall iguales a 93.75% y mejorados al 100% con la adición de una red GRNN final.
	 3. Y se verificó que esta red neuronal LVQ no posee mucha tolerancia al ruido al tener valores máximos de precisión y recall de 98.14% y 97.43% ante la aplicación de ruido a un nivel del 0.1% y valores mínimos de 90.25 y 88.05 respectivamente ante la aplicación del 10% de ruido.

-  [Vision computacional - Ruiz Gallo (2023)](https://hdl.handle.net/20.500.12893/11431) 
	- No abre xd

-  [Dirgido a escolares - UTP(2019)](https://alicia.concytec.gob.pe/vufind/Record/UTPD_fc14efa957d131c31225b23a723deaba/Description#tabnav)
	**Fundamentos**:
	- Cámaras RGB-D.
	- Procesamiento de imágenes, redes neuronales y deep learning.
	- Comunicación inalámbrica.

	**Proceso**
		1. Fotos
		2. Deteccion manual
		3. Segmentacion espacio-tiempo
		4. Extraccion car 3D
		5. Clasificacion con SVM

	Texto asociado al gesto. IOT, tiempo real.

- [Reconcimiento DL - UNSAAC (2017)](https://alicia.concytec.gob.pe/vufind/Record/RUNS_a282bd937cd1e41269096fa5c69e8a99/Description#tabnav)
	- Aprendizaje profundo  
	- Gestos de la mano  
	- Red neuronal convolucional  
	- Autoencoder apilado ruidoso  
	- LSP
	**ACCESO CERRADO**
### Teorica
- “Real time Finger Tracking and Contour Detection for Gesture Recognition using OpenCV” (R. Gurav, K. Kadbe, 2015)
	
	- Técnicas de preprocesamiento, con el fin de reconocer la región de la mano dentro de un fondo estático
	
- “Sign Language Conversion TooL (SLCTooL) Between 30 World Sign Languages” (Sastry A.S.C.S, et. al., 2018)
	 - Descriptor de imagen HOG, 
	 - eficiencia comparada con otros descriptores de imagen + algoritmo de clasificación
	 - 
-
- Hand gesture recognition using neural network based techniques” (Vladislava B., et. al., 2016)
	- Descriptor MLP

- IFT based approach on Bangla Sign Language Recognition” (Farhad Y., et. al, 2015)
	- Descriptor KNN


#### Upgrade
- [Gestos Dinamicos de brazos, con camaras de profundidad - UNMSM (2019)](https://alicia.concytec.gob.pe/vufind/Record/UNMS_1dc70ebec2e2441c6502d37a58278ab8)
- [Repo deteccion Manos y brazos](https://github.com/vladiH/peruvian_sign_language_translation)
#### Justificacion
- [Impacto y poco mas- UCV (2018)](# Aplicación móvil de interpretación del lenguaje de señas peruanas para discapacitados auditivos en la Asociación de Sordos de la Región Lima)
- [Minedu guia de aprendizaje (2015)](https://repositorio.minedu.gob.pe/handle/20.500.12799/5545)
## DataSets
- [Berru dataset UPC](https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet/tree/master) (Creative Commons)
-  [Aprendo en Casa](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/HDOAGH)
- [Videos Deletrando nombres PUCP](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/47OFDW)
- [Otro puc](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/OJYYYS)
- [Pos caras y manos etiquetadas en videos leng senas](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/OJYYYS)
## Tasks
- Crear un submodulo 
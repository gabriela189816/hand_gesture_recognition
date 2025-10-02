# HAND GESTURE DETECTION AND RECOGNITION
## Detección y reconocimiento de gestos de mano

**Objetivo**: Detectar y reconocer gestos de mano en 2D a partir de su forma comparándolos con gestos existentes en una base de datos. Implementar un algoritmo para la detección y reconocimiento de gestos basado en los *componentes principales (PCA)*.

**Artículo de referencia:**
> M. Turk, A. Pentland, Eigenfaces for Recognition, *Journal of Cognitive Neurosicence*, Vol. 3, No. 1, 1991, pp. 71-86. (Disponible en la página del curso)

En general, el método utiliza la **transformación de Karhunen-Loeve**. Cada cara almacenada en la base de datos es un vector de dimensión N. El *análisis de los componentes principales (PCA)* se utiliza para encontrar un subespacio de dimensión M cuyos vectores de base corresponden a las direcciones de máxima varianza en el espacio original de la imagen. Este nuevo subespacio es normalmente de dimensión más baja (M<<N).

-------------------------------------------------------------------------------------------------------------------------------------------------------
## RESULTADOS
**SET DE ENTRENAMIENTO**
<img src="Images/Training_set.png" alt="Set de entranamiento"/>

**PREPROCESAMIENTO DE LAS IMÁGENES**
<img src="Images/Preprocesamiento.png" alt="Preprocesamiento de las imágenes"/>

**MANO PROMEDIO**
<img src="Images/average_hand.png" alt="Mano promedio"/>

**PRIMERAS 10 EIGENHANDS MÁS REPRESENTATIVAS**
<img src="Images/FinalEigenHands.png" alt="Eigenhands"/>

**DEMOSTRACIÓN DE RECONOCIMIENTO**
<img src="Files/Experiment_demo.png" alt="Ejemplo de funcionamiento"/>

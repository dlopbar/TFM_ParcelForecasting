# Introducción

Actualmente nos encontramos en la era de la digitalización y la globalización, una era donde se genera una gran cantidad de información a diario que es almacenada por las empresas a conseguir ventajas competitivas para las empresas e incluso se está convirtiendo en una necesidad para conseguir mantenerse en el mercado.

Pero no sólo consiste en almacenar datos, consiste en almacenarlos, tratarlos, manipularlos, aprender de ellos, obtener resultados y actuar en consecuencia y de todo esto se encarga la ciencia de datos.

Entre las empresas dedicadas a la logística, encontramos aquellas que ofrecen servicios de envíos de paquetería. Para este tipo de empresas es vital el buen aprovechamiento de sus propios datos, concretamente predecir la actividad futura a corto, medio y largo plazo.

En el presente trabajo se centra el foco en el corto plazo, esto es, predecir cuántos paquetes se van a registrar el siguiente día en el sistema (lo cuáles serán desplazados el próximo día laborable). 

# Objetivo

El objetivo principal del trabajo es el de realizar una herramienta basada en analítica avanzada para la previsión del registro de paquetes el siguiente día en el sistema de una empresa de paquetería española. 
La obtención de dicho dato es importante para la empresa de cara a organizar los recursos necesarios tanto de personal como de transporte para el movimiento de los paquetes el día próximo debido a que la planificación se realiza con un día de antelación.

Concretamente, los objetivos del trabajo son los siguientes:
*   Obtención de la información, limpieza y tratamiento de los datos.
*   Análisis y visualización de los datos.
*   Implementación de modelos de previsión.
*   Evaluación de los modelos de previsión.
*   Interfaz web para la visualización y ejecución del modelo de previsión.

# Modelos

Concretamente, los modelos a testear son:
*   SARIMAX
*   PROPHET
*   RANDOM FOREST REGRESSOR
*   XGBOOST
*   LSTM

Se han encontrado diferentes escalas de resultados en los algoritmos tratados, encontrando algoritmos como Random Forest o XGBoost conocidos por su buen rendimiento, seguidos de LSTM que, actualmente con el desarrollo del Deep Learning, va ganando enteros y, por último, algoritmos estadísticos más tradicionales como SARIMAX o Prophet quedan a la cola.

Una vez elegido el modelo, se prepara para ser incluido en una interfaz web desarrollada con el módulo streamlit.

# Entorno

El repositorio (https://github.com/dlopbar/TFM_ParcelForecasting) contiene todo el código desarrollado para su ejecución. Sin embargo no contiene los datos, los cuales han de despositarse manualmente en la carpeta habilitada para ello.
En caso de necesitarlos, contactar con dlopbar416@gmail.com

La parte técnica del proyecto se ha desarrollado en un entorno virtual de Conda usando Visual Studio Code y las extensiones necesarias para Python y Jupyter Notebook.

Para replicar el entorno del proyecto y los módulos necesarios existe tanto el archivo “requirements.txt” como el “environment.yml”.

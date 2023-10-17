# Sistema web de detección de conducta criminal
El Sistema Web de Detección de Conducta Criminal es una aplicación innovadora desarrollada con Flask, que aprovecha las capacidades de TensorFlow y OpenCV para analizar y procesar imágenes en tiempo real con el propósito de identificar eventos y comportamientos sospechosos. Esta solución proporciona una herramienta valiosa para la seguridad y la vigilancia, permitiendo a las organizaciones y las autoridades detectar incidentes potencialmente críticos y tomar medidas preventivas de manera más efectiva.

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar la aplicación:

- Python 3.9


# Instalación
Para instalar la aplicacíon desde el repositorio debe seguir los siguientes pasos:

1. Instalar Miniconda ingresando al siguiente [enlace](https://docs.conda.io/projects/miniconda/en/latest/).

2. Crea un nuevo entorno en miniconda con Python 3.9 y activalo.
```bash
conda create -n tensorflow pip python=3.9
conda activate tensorflow
```


3. Instala los paquetes de flask y protobuf compatibles con la versión más reciente de la API de detección de objetos.
```bash
pip install PyYaml==5.3.1 protobuf==3.19.5 
```


4. Clona el repositorio en un directorio.
```bash
git clone https://github.com/tu-usuario/tu-repo.git
```


5. Ingresa a la carpeta del repositorio.


6. Instala Flask con el modulo de web sockets
```bash
pip install Flask gevent eventlet flask-socketio wmi
```


7. Instala las dependecias para la lectura de imagenes
```bash
pip install opencv-python mediapipe
```


8. Ejecuta el comando de arranque de la aplicación
```bash
flask --app main run --debug
```

# Cuentas de prueba

Correo electrónico: usuario_regula@email.com
Contraseña: patito
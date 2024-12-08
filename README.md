# Trabajos Prácticos de Robótica Móvil [en construcción]

## Instalación

Para poder ejecutar correctamente el código debe instalarse el módulo de este repositorio. Para ello, se debe instalar [Miniforge]() y crear un entorno virtual con `mamba`, instalar las dependencias y luego instalar el módulo en modo edición.

### Paso 1: Instalar Miniforge

Seguir las instrucciones de la [página oficial de Miniforge]() para instalar Miniforge en su sistema operativo.

### Paso 2: Instalar Mamba y entorno virtual

Una vez instalado Miniforge, abrir una terminal y ejecutar los siguientes comandos:

```bash
# Instalar mamba desde conda (luego de instalar Miniforge)
conda install mamba -c conda-forge
# Crear un entorno virtual con mamba
mamba create -n probotics python=3.11
# Activar el entorno virtual
mamba activate probotics
# Agregar los canales de conda-forge...
conda config --env --add channels conda-forge
# ... y robostack-staging
conda config --env --add channels robostack-staging
# Eliminar el canal defaults
conda config --env --remove channels defaults
```

### Paso 3: Instalar dependencias

Ahora instalamos ROS Noetic y las librerías necesarias de Python. Para ello, abrir una terminal y ejecutar los siguientes comandos:

```bash
# Install ros-noetic into the environment (ROS1)
mamba install ros-noetic-desktop
# Resetear el entorno para que ROS funcione
mamba deactivate
mamba activate probotics
# Instalar las dependencias de ROS
mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

En caso de usar Windows, Instalar Visual Studio 2017, 2019 or 2022 con el compilador de C++ (ver [este link](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-160) para más información). En caso de usar Linux o Mac OS, saltear este paso.
```bash
# Instalar Visual Studio command prompt (versión 2019):
mamba install vs2019_win-64
# Instalar Visual Studio command prompt (versión 2022):
mamba install vs2022_win-64
```

Ahora, instalar las dependencias de Python:
```bash
# Instalar el resto de dependencias
pip install -r requirements.txt
# Instalar el módulo en modo edición
pip install -e .
```

### Paso 4: Instalar FFmpeg

Para poder visualizar los videos generados en las primeras simulaciones es necesario instalar FFmpeg. Para ello, seguir las instrucciones de la [página oficial de FFmpeg](https://ffmpeg.org/download.html) correspondientes a su sistema operativo.

## Notebooks

La carpeta `notebooks` contiene los trabajos prácticos del curso.


## Uso

Para utilizar el módulo es posible ejecutar
```bash
python -m probotics <action> <arg1> <arg2> ...
```

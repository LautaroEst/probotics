# Trabajos Prácticos de Robótica Móvil [en construcción]

## Instalación

Para poder ejecutar correctamente el código debe instalarse el módulo de este repositorio. Para ello, se debe instalar [Miniforge]() y crear un entorno virtual con `mamba`, instalar las dependencias y luego instalar el módulo en modo edición:
```bash

## Instalar Mamba y entorno virtual

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

## Instalar dependencias

# Install ros-noetic into the environment (ROS1)
mamba install ros-noetic-desktop
# Resetear el entorno para que ROS funcione
mamba deactivate
mamba activate probotics
# Instalar las dependencias de ROS
mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

# En caso de usar Windows, Instalar Visual Studio 2017, 2019 or 2022 con el compilador de C++
# (ver https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-160 )
# Instalar Visual Studio command prompt (versión 2019):
# mamba install vs2019_win-64
# Instalar Visual Studio command prompt (versión 2022):
# mamba install vs2022_win-64

# Instalar el resto de dependencias
pip install -r requirements.txt
# Instalar el módulo en modo edición
pip install -e .
```

## Notebooks

La carpeta `notebooks` contiene los trabajos prácticos del curso.


## Uso

Para utilizar el módulo es posible ejecutar
```bash
python -m probotics <action> <arg1> <arg2> ...
```

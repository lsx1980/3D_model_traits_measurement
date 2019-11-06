BootStrap: docker
From: ubuntu:16.04

%labels
  Maintainer: Suxing Liu
  Version v1.01

%setup
  #----------------------------------------------------------------------
  # commands to be executed on host outside container during bootstrap
  #----------------------------------------------------------------------
  mkdir ${SINGULARITY_ROOTFS}/opt/code/

%files
  ./* /opt/code/

%post
  #----------------------------------------------------------
  # Install common dependencies and create default entrypoint,
  # commands to be executed inside container during bootstrap
  #----------------------------------------------------------
  # Install dependencies
  apt update
  apt install -y \
    build-essential \
    python3 \
    python-setuptools \
    python-numpy \
    python-matplotlib \
    ipython \
    ipython-notebook \
    python-pandas \
    python-sympy \
    python-nose \
    python-scipy \
    python-sklearn \
    python-numexpr \
    python-vtk \
    python-tk \
    python-wxgtk3.0 \
    gtk2-engines \
    gtk2-engines-* \
    overlay-scrollbar-gtk2 \
    unity-gtk-module-common \
    libcanberra-gtk-module \
    libatk-adaptor \
    libgail-common \
    xvfb \
    python-pip 

  #apt-get install --reinstall unity-gtk-module
  
  pip install --upgrade pip
  
  /usr/local/bin/pip install -U numpy 
  
  /usr/local/bin/pip install scikit-image \
                                rdp \
                                scikit-learn \
                                mayavi \
                                opencv-python \
                                openpyxl \
                                plyfile \
                                xvfbwrapper


  mkdir /lscratch /db /work /scratch
  
  chmod -R a+rwx /opt/code/
  
%environment
  #----------------------------------------------------------
  # Setup environment variables
  #----------------------------------------------------------
  PYTHONPATH=$PYTHONPATH:/opt/code/
  export PATH
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/code/
  export LD_LIBRARY_PATH

%runscript
  #----------------------------------------------------------
  # Run scripts inside container
  #----------------------------------------------------------
   # commands to be executed when the container runs
   echo "Arguments received: $*"
   exec /usr/bin/python "$@"
  
%test
  #----------------------------------------------------------
  # commands to be executed within container at close of bootstrap process
  #----------------------------------------------------------
   python --version
   #python requirement.py 
   

FROM ros:lunar

COPY . yamax/

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl openssh-server ca-certificates gnupg \
    && mkdir /var/run/sshd \
    && echo 'root:yamaximum314' | chpasswd \
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && echo 'X11UseLocalhost no' >> /etc/ssh/sshd_config \
    && touch ~/.Xauthority \
    && echo "deb http://packages.osrfoundation.org/gazebo/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-latest.list \
    && curl http://packages.osrfoundation.org/gazebo.key | apt-key add - \
    && apt-get update \
    && apt-get install -y gazebo7 libgazebo7-dev \
    && easy_install pip \
    && pip install tensorflow keras==2.0.6 keras-rl h5py gym \
    && apt-get update \
    && apt-get install -y --no-install-recommends xauth ros-lunar-gazebo-plugins ros-lunar-joint-state-publisher ros-lunar-rviz ros-lunar-robot-state-publisher ros-lunar-ros-control ros-lunar-ros-controllers ros-lunar-gazebo-ros ros-lunar-gazebo-ros-control ros-lunar-joint-state-controller ros-lunar-position-controllers \
    && git clone https://github.com/erlerobot/gym-gazebo.git \ 
    && cd gym-gazebo \
    && pip install -e . \
    && cd .. \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf yamax/devel yamax/build

WORKDIR /yamax

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

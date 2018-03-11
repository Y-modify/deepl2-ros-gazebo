FROM ubuntu:xenial

ENV ROS_DISTRO lunar

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Gazebo, buil
ADD gazebo-9.0.0-full+force_preserveWorldVelocity+warn.deb /tmp/gazebo.deb

RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb x11vnc fluxbox build-essential psmisc dirmngr curl ca-certificates gnupg software-properties-common \
    && echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116 \
    && echo "deb http://packages.osrfoundation.org/gazebo/ubuntu xenial main" > /etc/apt/sources.list.d/gazebo-latest.list \
    && curl http://packages.osrfoundation.org/gazebo.key | apt-key add - \
    && apt-add-repository -y ppa:libccd-debs \
    && apt-add-repository -y ppa:fcl-debs \
    && apt-add-repository -y ppa:dartsim \
    && apt-get update \
    && apt-get install --no-install-recommends -y  libdart6-dev libdart6-utils-urdf-dev python-rosdep python-rosinstall python-vcstools \
    && rosdep init \
    && rosdep update \
    && apt-get install --no-install-recommends -y ros-lunar-ros-core=1.3.1-0* ros-lunar-ros-base=1.3.1-0* \
    && easy_install pip \
    && pip install tensorflow==1.3.0 keras==2.0.6 keras-rl h5py gym
RUN apt-get install -y --no-install-recommends xauth ros-lunar-joint-state-publisher ros-lunar-rviz ros-lunar-robot-state-publisher ros-lunar-gazebo9-ros-pkgs ros-lunar-gazebo9-ros-control ros-lunar-ros-controllers ros-lunar-ros-control ros-lunar-joint-state-controller ros-lunar-position-controllers ros-lunar-xacro \
    && dpkg -r --force-depends gazebo9 libgazebo9 libgazebo9-dev \
    && dpkg -i --force-depends --force-overwrite /tmp/gazebo.deb \
    && git clone https://github.com/erlerobot/gym-gazebo.git \
    && cd gym-gazebo \
    && pip install -e . \
    && cd .. \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY :1

COPY . yamax/
RUN rm -rf yamax/devel yamax/build \
    && . /opt/ros/lunar/setup.sh \
    && cd yamax \
    && catkin_make \
    && echo '. /opt/ros/lunar/setup.sh' >> /etc/profile \
    && echo '. /yamax/devel/setup.sh' >> /etc/profile

WORKDIR /yamax

COPY ./vnc-startup.sh /
EXPOSE 5900

CMD bash -i -c "/vnc-startup.sh && roslaunch yamax_gazebo world.launch gui:=True headless:=False"

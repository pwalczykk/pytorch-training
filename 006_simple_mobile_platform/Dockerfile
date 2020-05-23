FROM ros:melodic-ros-base

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install -y \
    ros-melodic-urdf \
    ros-melodic-xacro \
    ros-melodic-tf \
    ros-melodic-rviz \
    ros-melodic-robot-state-publisher

RUN apt-get update && apt-get install -y \
    python-pip

RUN pip install \
    numpy \
    pybullet \
    pytest

COPY ./src /root/src

WORKDIR /root/src

#!/bin/bash

trap "xhost -local:root" EXIT
xhost +local:root

docker run \
  -it \
  --runtime=nvidia \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  pwalczykk/006_simple_mobile_platform:latest \
  /bin/bash

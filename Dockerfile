FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ECE3574_REFERENCE_ENV="Y"

RUN apt-get update && apt-get upgrade
RUN apt-get -y install build-essential coreutils cmake
RUN apt-get -y install python3-pip python3-dev

# Usage: docker run -it --mount type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
# Usage with x11-forwarding: docker run -it -e DISPLAY=docker.for.mac.host.internal:0 --mount 
#                            type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install python3-pip python3-dev
RUN apt-get -y install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
RUN apt-get -y install ffmpeg

# upgrade build tools before installing dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install pyaudio

# Usage: docker run -it --mount type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
# Usage with x11-forwarding: docker run -it -e DISPLAY=docker.for.mac.host.internal:0 --mount 
#                            type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
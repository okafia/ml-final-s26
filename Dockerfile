FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ECE3574_REFERENCE_ENV="Y"

RUN apt-get update && apt-get upgrade
RUN apt-get -y install python3-pip python3-dev
RUN apt-get -y install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
RUN apt-get -y install ffmpeg libav-tools
RUN pip install pyaudio

# Usage: docker run -it --mount type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
# Usage with x11-forwarding: docker run -it -e DISPLAY=docker.for.mac.host.internal:0 --mount 
#                            type=bind,src=$PWD,dst=/mnt [YOUR_NAME]/ref-env
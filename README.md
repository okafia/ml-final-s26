# Drum and bass auto-DJ
_Python3 version._

This repository contains an adaptation of the automatic DJ system developed by Len Vande Veire, under the supervision of prof. Tijl De Bie. It has been updated to implement K nearest neighbors and K means clustering for adaptive track selection.

The system is described in more detail in the paper (link)

## Installation

The auto-DJ system has been tested for Ubuntu 18.04 LTS. Use the provided Dockerfile to build and run the system in a container environment.

```
# build docker container
docker build -t ref-env .

# run the container
docker run -it --mount type=bind,src=$PWD,dst=/mnt ref-env

# inside the container, install dependencies and build project
cd /mnt
pip3 install .
# a build folder will be generated

```

## Running the application

Run the application in the build folder with the following command:

`python -m autodj.main`

The application is controlled using commands. A typical usage would be as follows:

```
$ python -m autodj.main

>> loaddir /home/username/music/drumandbass
Loading directory "/home/username/music/drumandbass"...
175 songs loaded (0 annotated).
>> annotate
Annotating music in song collection...
...
Done annotating!
>> play
Started playback!
```


The following commands are available:

* `loaddir <directory>` : Add the _.wav_ and _.mp3_ audio files in the specified directory to the pool of available songs.
* `annotate` : Annotate all the files in the pool of available songs that are not annotated yet. Note that this might take a while, and that in the current prototype this can only be interrupted by forcefully exiting the program (using the key combination `Ctrl+C`).
* `play` : Start a DJ mix. This command must be called after using the `loaddir` command on at least one directory with some annotated songs. Also used to continue playing after pausing.
* `play save`: Start a DJ mix, and save it to disk afterwards.
* `pause` : Pause the DJ mix.
* `stop` : Stop the DJ mix.
* `skip` : Skip to the next important boundary in the mix. This skips to either the beginning of the next crossfade, the switch point of the current crossfade or the end of the current crossfade, whichever comes first.
* `s` : Shorthand for the skip command
* `showannotated` : Shows how many of the loaded songs are annotated.
* `debug` : Toggle debug information output. This command must be used before starting playback, or it will have no effect.
* `stereo` : Toggle stereo audio support (enabled by default). Note: stereo audio is an experimental feature and leads to a longer processing time per crossfade.

To exit the application, use the `Ctrl+C` key combination.

## Changes in the Python3 version

The Python3 version of the auto-DJ system features the same functionality as the original prototype.
The main changes in the code base are:

* Stereo audio support (experimental, enabled by default).
* All annotations are saved in a single .json file per song.
* Code refactoring: the annotation modules are now in a separate subpackage, and are incorporated into the auto-DJ application using wrapper classes.


## Copyright information

Released under AGPLv3 license.

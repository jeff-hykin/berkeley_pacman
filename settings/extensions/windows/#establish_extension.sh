#!/usr/bin/env bash

inside_wsl=""
# check if file exists
if [ -f "$FORNIX_FOLDER/settings/.cache/env_is_inside_wsl" ]
then
    inside_wsl="true"
fi
# if WSL_DISTRO_NAME exists
if [ -n "$WSL_DISTRO_NAME" ]
then
    inside_wsl="true"
    touch "$FORNIX_FOLDER/settings/.cache/env_is_inside_wsl"
fi


# if inside WSL
if [ -n "$wsl_version" ]
then
    # set the display port for tkinter
    export DISPLAY="$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')":0.0
fi
#!/bin/sh

Xvfb $DISPLAY -screen 0 1280x720x24 &
fluxbox &
x11vnc -display $DISPLAY -nopw -xkb

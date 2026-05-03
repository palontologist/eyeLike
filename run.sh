#!/usr/bin/env bash
# eyeLike launcher with X11 backend for Wayland compatibility

cd /home/palontologist/Downloads/dev/eyeLike/build

# Use X11 backend for GTK2 compatibility on Wayland
export GDK_BACKEND=x11

# Launch with OpenCV GTK2 support
exec nix-shell -p "(opencv4.override { enableGtk2 = true; })" gtk2 pkg-config --run "./bin/eyeLike"

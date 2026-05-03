{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (opencv4.override {
      enableGtk2 = true;
      enableFfmpeg = true;
      enableGStreamer = true;
    })
    gtk2
    pkg-config
    cmake
    gcc
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    glib
    ffmpeg
    libpng
    libjpeg
    libtiff
    libwebp
    openjpeg
    openexr
    hdf5
    openblas
    ocl-icd
    zlib
  ];
  
  shellHook = ''
    echo "OpenCV with GTK support enabled"
    echo "Run: cd build && cmake .. && make && ./bin/eyeLike"
  '';
}

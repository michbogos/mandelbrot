# Mandelbrot set
![mandelbrot zoom](https://github.com/michbogos/mandelbrot/blob/main/zoom.gif?raw=true)

This is an experiment in rendering the mandelbrot set and fractals in general

Zoom generated with numpy python implementation

## Multithreaded c++ implementation

Compile with

``` bash

make multithreaded

```

This is fast enough to generate hundreds of frames.

Uses stb_image_write.h to write frames. Frames are saved in frames/*.png.

Currently the fastest high precision implementation

## Optimized c implementation

Compile with

``` bash

make optimized

```

## Interactive shader raylib c implementation
![mandelbrot image](https://github.com/michbogos/mandelbrot/blob/main/mandelbrot.png?raw=true)

Compile with

``` bash

make raylib

```

Requires raylib
Use the Arrow keys to move. Space to zoom.

## Numpy implementation

Run main.py with python.

Requires numpy
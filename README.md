# Mandelbrot set
![mandelbrot zoom](https://github.com/michbogos/mandelbrot/blob/main/zoom.gif?raw=true)

This is an experiment in rendering the mandelbrot set and fractals in general

Zoom generated with numpy python implementation

## Interactive shader raylib c implementation
![mandelbrot image](https://github.com/michbogos/mandelbrot/blob/main/mandelbrot.png?raw=true)

Compile with

``` bash

gcc main.c -Ofast -o mandelbrot -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

```

Requires raylib

Use the Arrow keys to move. Space to zoom.
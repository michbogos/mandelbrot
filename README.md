# Mandelbrot set

![mandelbrot image](https://github.com/michbogos/mandelbrot/blob/main/mandelbrot.png?raw=true)

This is an experiment in rendering the mandelbrot set and fractals in general

Compile with

``` bash

gcc main.c -Ofast -o mandelbrot -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

```

Requires raylib to be installed

Use the Arrow keys to move. Space to zoom.
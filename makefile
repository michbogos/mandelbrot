optimized: optimized.c
	gcc optimized.c -Ofast -o optimized -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL -lm

multithreaded: multithreaded.cpp
	g++ multithreaded.cpp -o multithreaded -lm -Ofast -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

raylib: main.c
	gcc main.c -Ofast -o mandelbrot -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
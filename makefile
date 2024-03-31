optimized: optimized.c
	gcc optimized.c -Ofast -o optimized -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL -lm

multithreaded: multithreaded.cpp
	g++ multithreaded.cpp -o multithreaded -lm -Ofast -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

raylib: main.c
	gcc main.c -Ofast -o mandelbrot -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

gmp: gmp.cpp
	g++ gmp.cpp -o gmp -lm -Ofast -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

avx: avx.cpp
	g++ avx.cpp -o avx -lm -march=native -mtune=native -Ofast  -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

hip: hip.cpp
	hipcc hip.cpp -o hip -lm -O3
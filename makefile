optimized: impl_misc/optimized.c
	gcc impl_misc/optimized.c -Ofast -o optimized -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL -lm

multithreaded: impl_misc/multithreaded.cpp
	g++ impl_misc/multithreaded.cpp -o multithreaded -lm -Ofast -mtune=native -march=native -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

raylib: impl_glsl/main.c
	gcc impl_glsl/main.c -Ofast -o mandelbrot -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

gmp: impl_misc/gmp.cpp
	g++ impl_misc/gmp.cpp -o program -lm -lgmp -march=native -mtune=native -Ofast  -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

avx: impl_misc/avx.cpp
	g++ impl_misc/avx.cpp -o avx -lm -march=native -mtune=native -Ofast  -fno-signed-zeros -fno-trapping-math -fopenmp -D_GLIBCXX_PARALLEL

hip: impl_gpu/hip.cpp
	hipcc impl_gpu/hip.cpp -o hip -lm -O3
# Calcul de l'ensemble de Mandelbrot en parallèle

## Dépendances

Ce projet a besoin de SDL2 pour la visualisation
```bash
apt-get install libSDL2-dev
```

De MPI pour le calcul en parallèle sur CPU:
```bash
apt-get install openmpi-bin
```

De CUDA pour le calcul en parallèle sur GPU:
```bash
apt-get install nvidia-cuda-toolkit
```

## Compiler et lancer le programme

Pour visualiser l'ensemble de Mandelbrot sur CPU (MPI) :
```bash
mpicc mandelbrot_cpu.c -o mandelbrot_cpu.o -lSDL2
mpirun -np <nb de coeurs> mandelbrot_cpu.o
```
		
Pour visualiser l'ensemble de Mandelbrot sur GPU (CUDA) :
```bash
nvcc mandelbrot_gpu.cu -o mandelbrot_gpu.o -lSDL2 -lcublas
./mandelbrot_gpu.o
```

Ou bien 
```bash
nvcc mandelbrot_gpu.cu -lSDL2 -lcublas --run
```

## Utilisation du programme 

Pour zoomer dans l'ensemble : viser un point avec le curseur de la souris et clic gauche pour zoomer à l'endroit 

Pour revenir à la vue de l'ensemble de départ : clic droit

Pour passer entre l'ensemble de Mandelbrot et de Julia, appuyer sur une touche de clavier

En mode 'Julia' : maintenir le clic gauche, l'ensemble de Julia sera calculer au point où se situe le curseur (et dépend du zoom)
	

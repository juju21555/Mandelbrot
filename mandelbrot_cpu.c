#include "mpi.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <SDL2/SDL.h>

#define SIZE   1024

unsigned char pixels[SIZE * SIZE * 3];

float R_F2(float a, float r, float c){
    return a + r*r - c*c ;
}

float I_F2(float b, float r, float c){
    return b + 2*r*c;
}

char characterGrayScale(int grayScale)
{
    if (grayScale < 25) return ' ';
    if (grayScale < 50) return '.';
    if (grayScale < 75) return ':';
    if (grayScale < 100) return '-';
    if (grayScale < 125) return '=';
    if (grayScale < 150) return '+';
    if (grayScale < 175) return '*';
    if (grayScale < 200) return '#';
    if (grayScale < 225) return '%';
    return '@';
}




unsigned char getColorR(int t){
    int t_i = (t/60) % 6;
    float f = t/60.0 - t_i;
    float s = 1.0;
    float v = 1-t/255;
    int8_t l = v * (1-s) * 255;
    int8_t m = v * (1-f*s) * 255;
    int8_t n = v * (1-(1-f)*s) * 255;
    switch (t_i){
        case 0:
            return v*255;
        case 1:
            return m;
        case 2:
            return l;
        case 3:
            return l;
        case 4:
            return n;
        case 5:
            return v*255;
    }
}

unsigned char getColorG(int t){
    int t_i = (t/60) % 6;
    float f = t/60.0 - t_i;
    float s = 1.0;
    float v = 1-t/255;
    int8_t l = v * (1-s) * 255;
    int8_t m = v * (1-f*s) * 255;
    int8_t n = v * (1-(1-f)*s) * 255;
    switch (t_i){
        case 0:
            return n;
        case 1:
            return v*255;
        case 2:
            return v*255;
        case 3:
            return m;
        case 4:
            return l;
        case 5:
            return l;
    }
}

unsigned char getColorB(int t){
    int t_i = (t/60) % 6;
    float f = t/60.0 - t_i;
    float s = 1.0;
    float v = 1-t/255;
    int8_t l = v * (1-s) * 255;
    int8_t m = v * (1-f*s) * 255;
    int8_t n = v * (1-(1-f)*s) * 255;
    switch (t_i){
        case 0:
            return l;
        case 1:
            return l;
        case 2:
            return n;
        case 3:
            return v*255;
        case 4:
            return v*255;
        case 5:
            return m;
    }
}


void render(SDL_Renderer*  pRenderer, int * T_ALL){

    for (int i=0; i < SIZE; i++){
        for (int j=0; j < SIZE; j++){
            pixels[(i*SIZE+j)*3]      = getColorR(T_ALL[j*SIZE+i]);
            pixels[(i*SIZE+j)*3 + 1]  = getColorG(T_ALL[j*SIZE+i]);
            pixels[(i*SIZE+j)*3 + 2]  = getColorB(T_ALL[j*SIZE+i]);
            if (T_ALL[j*SIZE+i] < 0){
                pixels[(i*SIZE+j)*3]      = getColorR(255);
                pixels[(i*SIZE+j)*3 + 1]  = getColorG(255);
                pixels[(i*SIZE+j)*3 + 2]  = getColorB(255);
            }
        }
    }


    SDL_Rect texture_rect;
    texture_rect.x=0; texture_rect.y=0; texture_rect.w=SIZE; texture_rect.h=SIZE;
    SDL_Surface *surface = SDL_CreateRGBSurfaceFrom((void*)pixels,
                    SIZE,
                    SIZE,
                    3 * 8,                  // bits per pixel = 24
                    SIZE * 3,               // pitch
                    0x0000FF,               // red mask
                    0x00FF00,               // green mask
                    0xFF0000,               // blue mask
                    0);                     // alpha mask (none)

    SDL_Texture *texture = SDL_CreateTextureFromSurface(pRenderer, surface);

    SDL_RenderClear(pRenderer);
    SDL_RenderCopy(pRenderer, texture, NULL, &texture_rect);
    SDL_RenderPresent(pRenderer);
}


void MPI_CALC_Mandelbrot_Spiral(double center_x, double center_y, double size, int rank, int ROOT, int TOTAL_CORE, int * T, int * T_ALL, int * BUFF_X, int * BUFF_Y){
    
    double re, im, tmp;
    int lastIdx = 0;

    int n, i, j, ic;
    double c_r, c_i;
    
    for (int i=0; i<SIZE/TOTAL_CORE; i++){
        for (int j=0; j<SIZE; j++){
            T[i*SIZE+j] = -2;
        }
    }

    for (int i_core=0; i_core<TOTAL_CORE; i_core++){

        if (rank == i_core){

            j = 0;
            for (i = 0; i < SIZE/TOTAL_CORE; i++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }

            i = 0;
            for (j = 0; j < SIZE; j++){ 
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }    

            j = SIZE - 1;
            for (i = 0; i < SIZE/TOTAL_CORE; i++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }

            i = SIZE/TOTAL_CORE - 1;
            for (j = 0; j < SIZE; j++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }
            
            while (lastIdx > 0){
                lastIdx--;
                i = BUFF_X[lastIdx] + i_core*(SIZE/TOTAL_CORE);
                ic = BUFF_X[lastIdx];
                j = BUFF_Y[lastIdx];

                n = 0;
                c_r = size*(i*1./SIZE-0.5) + center_x;
                c_i = size*(j*1./SIZE-0.5) + center_y;
                re = c_r;
                im = c_i;

                while (n < 255 && (re*re + im*im < 4)){
                    tmp = R_F2(c_r, re, im);
                    im = I_F2(c_i, re, im);
                    re = tmp;
                    n++;
                }

                T[ic*SIZE+j] = n;


                if (n < 255){
                    for (int u=-1; u<=1; u++){
                        for (int v=-1; v<=1; v++){
                            if (0 <= ic+u && ic+u < SIZE/TOTAL_CORE && 0 <= j+v && j+v < SIZE){
                                if (T[(ic+u)*SIZE+j+v] == -2){
                                    BUFF_X[lastIdx] = ic+u;
                                    BUFF_Y[lastIdx] = j+v;
                                    T[(ic+u)*SIZE+j+v] = -1;
                                    lastIdx++;
                                }
                            }
                        }
                    }
                }
            }



        }
    }

    MPI_Gather( T,      (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // SENDER
                T_ALL,  (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // RECEIVER
                ROOT,   MPI_COMM_WORLD);
}

// Calcul de l'ensemble de Mandelbrot en parallèle efficace (car l'ensemble est compact, on ne calcul que les bords de l'ensemble)
void MPI_CALC_Julia_Spiral(double center_x, double center_y, double size, int rank, int ROOT, int TOTAL_CORE, int * T, int * T_ALL, int * BUFF_X, int * BUFF_Y){
    
    double re, im, tmp;
    int lastIdx = 0;

    int n, i, j, ic;
    double c_r, c_i;
    
    for (int i=0; i<SIZE/TOTAL_CORE; i++){
        for (int j=0; j<SIZE; j++){
            T[i*SIZE+j] = -2;
        }
    }

    for (int i_core=0; i_core<TOTAL_CORE; i_core++){

        if (rank == i_core){

            // On initialise les bords de l'ensemble et on ajoute les points aux alentours dans une queue pour qu'ils soient calculé à l'étape suivante

            j = 0;
            for (i = 0; i < SIZE/TOTAL_CORE; i++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }

            i = 0;
            for (j = 0; j < SIZE; j++){ 
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }    

            j = SIZE - 1;
            for (i = 0; i < SIZE/TOTAL_CORE; i++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }

            i = SIZE/TOTAL_CORE - 1;
            for (j = 0; j < SIZE; j++){
                BUFF_X[lastIdx] = i;
                BUFF_Y[lastIdx] = j;
                T[i*SIZE+j] = -1;
                lastIdx++;
            }
            
            // Tant qu'on a des points dans la queue, on les calculs puis on ajoute les points aux alentours dans la queue, si un point atteint l'itération maximum
            // il est considéré comme un bord, et on n'ajoute pas les points alentours dans la queue
            while (lastIdx > 0){ 
                lastIdx--;
                i = BUFF_X[lastIdx] + i_core*(SIZE/TOTAL_CORE);
                ic = BUFF_X[lastIdx];
                j = BUFF_Y[lastIdx];

                n = 0;
                c_r =  center_x;
                c_i =  center_y;
                re = size*(i*1./SIZE-0.5);
                im = size*(j*1./SIZE-0.5);

                while (n < 255 && (re*re + im*im < 4)){
                    tmp = R_F2(c_r, re, im);
                    im = I_F2(c_i, re, im);
                    re = tmp;
                    n++;
                }

                T[ic*SIZE+j] = n;

                if (n < 255){
                    for (int u=-1; u<=1; u++){
                        for (int v=-1; v<=1; v++){
                            if (0 <= ic+u && ic+u < SIZE/TOTAL_CORE && 0 <= j+v && j+v < SIZE){
                                if (T[(ic+u)*SIZE+j+v] == -2){
                                    BUFF_X[lastIdx] = ic+u;
                                    BUFF_Y[lastIdx] = j+v;
                                    T[(ic+u)*SIZE+j+v] = -1;
                                    lastIdx++;
                                }
                            }
                        }
                    }
                }
            }



        }
    }

    MPI_Gather( T,      (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // SENDER
                T_ALL,  (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // RECEIVER
                ROOT,   MPI_COMM_WORLD);
    

}

// Calcul de l'ensemble de Mandelbrot en parallèle naïf
void MPI_CALC_Mandelbrot(double center_x, double center_y, double size, int rank, int ROOT, int TOTAL_CORE, int * T, int * T_ALL){
    
    double re, im, tmp;

    int n;
    double c_r, c_i;

    for (int i_core=0; i_core<TOTAL_CORE; i_core++){

        if (rank == i_core){

            for (int i = 0; i < SIZE/TOTAL_CORE; i++){
                for (int j = 0; j < SIZE; j++){
                    n = 0;
                    c_r = size*(i*1./SIZE-0.5) + center_x + i_core * size/TOTAL_CORE;
                    c_i = size*(j*1./SIZE-0.5) + center_y;
                    re = c_r;
                    im = c_i;

                    while (n < 255 && (re*re + im*im < 4)){
                        tmp = R_F2(c_r, re, im);
                        im = I_F2(c_i, re, im);
                        re = tmp;
                        n++;
                    }

                    T[i*SIZE+j] = n;
                }    
            }

        }
    }

    MPI_Gather( T,      (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // SENDER
                T_ALL,  (SIZE/TOTAL_CORE) * SIZE,      MPI_INT,        // RECEIVER
                ROOT,   MPI_COMM_WORLD);
    

}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int buf;
    int ROOT = 0;
    int numtasks, rank, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    int * T, * T_ALL, * BUFF_X, * BUFF_Y;

    
    clock_t t1, t2, t;


    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD ,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD ,&rank);
    MPI_Get_processor_name(hostname, &len);

    printf("NOMBRE DE COEURS UTILISES : %i\n", numtasks);


    T_ALL   = (int*)malloc(SIZE*SIZE*sizeof(int));              // -> Matrice représentant l'ensemble de Mandelbrot 
    T       = (int*)malloc((SIZE/numtasks)*SIZE*sizeof(int));   // -> Partie de la matrice représentant l'ensemble de Mandelbrot calculer sur un seul coeur
    BUFF_X  = (int*)malloc((SIZE/numtasks)*SIZE*sizeof(int));   // -> Liste simulant une queue utilisé pour calculer Mandelbrot de manière efficace
    BUFF_Y  = (int*)malloc((SIZE/numtasks)*SIZE*sizeof(int));   // -> Liste simulant une queue utilisé pour calculer Mandelbrot de manière efficace
    int lastIdx = 0;


    double center_x = 0., center_y = 0., size = 3.;
    double data[3];
    data[0] = 0.; // center x
    data[1] = 0.; // center y
    data[2] = 3.; // size
    data[3] = 0.;  // 0 -> Mandelbrot, 1 -> Julia, Autre -> Fin du programme

    SDL_Window* pWindow = NULL;      
    SDL_Renderer* pRenderer = NULL;   

    if (rank==ROOT){

        if(SDL_Init(SDL_INIT_VIDEO) < 0){
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "[debug] %s", SDL_GetError());
            return 0;
        }
        

        pWindow = SDL_CreateWindow("SDL Programme", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SIZE, SIZE, SDL_WINDOW_SHOWN);       
        if (pWindow == NULL){         
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "[DEBUG] > %s", SDL_GetError());         
            SDL_Quit();         
            return 0;     
        }       
        
        pRenderer = SDL_CreateRenderer(pWindow, -1, SDL_RENDERER_ACCELERATED);       
        if (pRenderer == NULL){         
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "[DEBUG] > %s", SDL_GetError());         
            SDL_Quit();         
            return 0;     
        }
    }

    SDL_Event events;
    int xMouse, yMouse;

    unsigned hold=0, isOpen=1, isMandelbrot=1;
    float cx, cy;

    MPI_CALC_Mandelbrot_Spiral(data[0], data[1], data[2], rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
    render(pRenderer, T_ALL);

    while (isOpen) {
        if (rank!=ROOT){
            MPI_Bcast(data, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
            if (data[3]==0.){
                MPI_CALC_Mandelbrot_Spiral(data[0], data[1], data[2], rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
            }
            else if (data[3]==1.){
                MPI_CALC_Julia_Spiral(data[0], data[1], 3., rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
            }
            else{
                isOpen = 0;
            }
        }

        else{
  	        if (hold && !isMandelbrot){
                SDL_GetMouseState(&xMouse,&yMouse);
                cx = data[0];
                cy = data[1];
                data[0] = data[2]*(xMouse*1.0/SIZE - 0.5) + data[0];
                data[1] = data[2]*(yMouse*1.0/SIZE - 0.5) + data[1];
                t1 = clock();
                MPI_Bcast(data, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                MPI_CALC_Julia_Spiral(data[0], data[1], 3., rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
                t2 = clock();
                t = t2-t1;
                printf("Temps Julia MPI : %i ms\n", (int) (t * 1000 / CLOCKS_PER_SEC));
                render(pRenderer, T_ALL);
                data[0] = cx;
                data[1] = cy;
            }
            while (SDL_PollEvent(&events)) {
                switch (events.type) {
                    case SDL_QUIT:
                        isOpen = 0;
                        break;

                    case SDL_KEYDOWN:
                        isMandelbrot = 1 - isMandelbrot;
                        break;

                    case SDL_MOUSEBUTTONDOWN:
                        if (isMandelbrot){
                            SDL_GetMouseState(&xMouse,&yMouse);
                            data[2] /= 2;
                            data[0] = data[2]*(xMouse*1.0/SIZE - 0.5)+data[0];
                            data[1] = data[2]*(yMouse*1.0/SIZE - 0.5)+data[1];

                            if (events.button.button == SDL_BUTTON_RIGHT){
                                data[2] = 3.;
                                data[0] = 0.;
                                data[1] = 0.;
                            }

                            t1 = clock();
                            MPI_Bcast(data, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                            
                            MPI_CALC_Mandelbrot_Spiral(data[0], data[1], data[2], rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
                            t2 = clock();
                            t = t2-t1;
                            printf("Temps Mandelbrot MPI : %i ms\n", (int) (t * 1000 / CLOCKS_PER_SEC));
                            render(pRenderer, T_ALL);
                        }
                        else{
                            data[3] = 1. - data[3];
                            hold = 1;
                        }
                        break;

                    case SDL_MOUSEBUTTONUP:
                        if (!isMandelbrot){
                            data[3] = 1. - data[3];
                            MPI_Bcast(data, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
                            MPI_CALC_Mandelbrot_Spiral(data[0], data[1], data[2], rank, ROOT, numtasks, T, T_ALL, BUFF_X, BUFF_Y);
                            render(pRenderer, T_ALL);
                            hold = 0;
                        }
                        break;
                }
            }
        }
    }

    if (rank==ROOT){
        data[3] = -1.;
        MPI_Bcast(data, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        SDL_DestroyRenderer(pRenderer);
        SDL_DestroyWindow(pWindow); 
        SDL_Quit();
    }

    free(T);
    free(T_ALL);
    free(BUFF_X);
    free(BUFF_Y);

    MPI_Finalize();

    return 0;
}

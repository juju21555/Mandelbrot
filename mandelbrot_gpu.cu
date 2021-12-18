#include <stdio.h>
#include <SDL2/SDL.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#undef main

#define SIZE   1024

unsigned char pixels[SIZE * SIZE * 3];

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
    return 0;
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
    return 0;
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
    return 0;
}


void render(SDL_Renderer*  pRenderer, int *T_ALL){

    for (int i=0; i < SIZE; i++){
        for (int j=0; j < SIZE; j++){
            pixels[(j*SIZE+i)*3]      = getColorR(T_ALL[(j*SIZE+i)]);
            pixels[(j*SIZE+i)*3 + 1]  = getColorG(T_ALL[(j*SIZE+i)]);
            pixels[(j*SIZE+i)*3 + 2]  = getColorB(T_ALL[(j*SIZE+i)]);
            if (T_ALL[(j*SIZE+i)] < 0){
                pixels[(j*SIZE+i)*3]      = getColorR(255);
                pixels[(j*SIZE+i)*3 + 1]  = getColorG(255);
                pixels[(j*SIZE+i)*3 + 2]  = getColorB(255);
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


__global__ void JULIA_GPU(int *T, double size, double centerX, double centerY){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  double c_r = centerX;
  double c_c = centerY;
  double z_r = size*(i*1.0/SIZE - 0.5);
  double z_c = size*(j*1.0/SIZE - 0.5);
  double tmp = 0.0;
  int n = 0;
  while (n < 255 && (z_r*z_r + z_c*z_c) < 4){
    tmp = z_r*z_r - z_c*z_c + c_r;
    z_c = 2 * z_r*z_c + c_c;
    z_r = tmp;
    n++;
  }
  T[j*SIZE+i] = n;
}

__global__ void MANDELBROT_GPU(int *T, double size, double centerX, double centerY){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  double c_r = size*(i*1.0/SIZE - 0.5) + centerX;
  double c_c = size*(j*1.0/SIZE - 0.5) + centerY;
  double z_r = 0.0, tmp = 0.0, z_c = 0.0;
  int n = 0;
  while (n < 255 && (z_r*z_r + z_c*z_c) < 4){
    tmp = z_r*z_r - z_c*z_c + c_r;
    z_c = 2 * z_r*z_c + c_c;
    z_r = tmp;
    n++;
  }
  T[j*SIZE+i] = n;
}


int main(void)
{
  int *T, *d_T;


  dim3 numBlocks(32, 32);
  dim3 threadsPerBlock(32, 32);

  T = (int*)malloc(SIZE*SIZE*sizeof(int));

  cudaMalloc(&d_T, SIZE*SIZE*sizeof(int));

  for (int i = 0; i < SIZE*SIZE; i++) {
    T[i] = 1;
  }

  cudaMemcpy(d_T, T, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);

  MANDELBROT_GPU<<<numBlocks, threadsPerBlock>>>(d_T, 3.0, 0.0, 0.0);

  cudaMemcpy(T, d_T, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);




  SDL_Window* pWindow = NULL;      
  SDL_Renderer* pRenderer = NULL;   

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

  SDL_Event events;
  int xMouse, yMouse;
  double cx, cy, size=3.0, old_cx=0.0, old_cy=0.0;

  render(pRenderer, T);

  unsigned hold=0, isOpen=1, isMandelbrot=1;

  while (isOpen){
  	if (hold && !isMandelbrot){
          SDL_GetMouseState(&xMouse,&yMouse);
	      	cx = size*(xMouse*1.0/SIZE - 0.5) + old_cx;
	      	cy = size*(yMouse*1.0/SIZE - 0.5) + old_cy;
  	
          JULIA_GPU<<<numBlocks, threadsPerBlock>>>(d_T, 3., cx, cy);
          cudaMemcpy(T, d_T, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
          render(pRenderer, T);
  	}
    while (SDL_PollEvent(&events)) {
      switch (events.type) {
          case SDL_QUIT:
            isOpen = 0;
            break;

          case SDL_KEYDOWN:
            isMandelbrot = 1-isMandelbrot;
            break;

          case SDL_MOUSEBUTTONDOWN:
            if (isMandelbrot){
              SDL_GetMouseState(&xMouse,&yMouse);
              size /= 2;
	      	    cx = size*(xMouse*1.0/SIZE - 0.5)+old_cx;
	      	    cy = size*(yMouse*1.0/SIZE - 0.5)+old_cy;

              if (events.button.button == SDL_BUTTON_RIGHT){
                size  = 3.;
                cx    = 0.;
                cy    = 0.;
              }

              MANDELBROT_GPU<<<numBlocks, threadsPerBlock>>>(d_T, size, cx, cy);
              cudaMemcpy(T, d_T, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
              render(pRenderer, T);

              old_cx = cx;
              old_cy = cy;
            }
            else{
              hold = 1;
            }
            break;

          case SDL_MOUSEBUTTONUP:
            if (!isMandelbrot){
              MANDELBROT_GPU<<<numBlocks, threadsPerBlock>>>(d_T, size, old_cx, old_cy);
              cudaMemcpy(T, d_T, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
              render(pRenderer, T);
              hold = 0;
            }
          	break;
      }
    }
  }

  SDL_DestroyRenderer(pRenderer);
  SDL_DestroyWindow(pWindow); 
  SDL_Quit();

  cudaFree(d_T);
  free(T);
}

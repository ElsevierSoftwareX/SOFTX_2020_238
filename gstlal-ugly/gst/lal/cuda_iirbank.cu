extern "C"
{
#include "cuda_iirbank.h"
}
int cudaUtils()
{
  printf("nothing\n");
  char *data;
  char *cudaData;
  data = (char*) malloc (12 * sizeof(char));
  strncpy(data, "hello world!", 12);
  data[11] = '\0'; 
  data[0] = 'b';

  cudaMalloc( (void **)&cudaData, 12 * sizeof(char) );
  cudaMemcpy( cudaData, data, 12 * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy( data, cudaData, 12 * sizeof(char), cudaMemcpyDeviceToHost);
  printf("%s\n", data);
	
	return 0;
}

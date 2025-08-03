#include<cuda_runtime.h>
#include<iostream>
#include <cstdlib>
using namespace std;
#define N 100
__global__ void Add(float *a,float *b,float *c, int N){
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id <N){
		c[id]=a[id]+b[id];
	}
} 

int main(){
	float *h_a=(float*)malloc(N*sizeof(float));
	float *h_b=(float*)malloc(N*sizeof(float));
	float *h_c =(float)malloc(N*sizeof(float));
	size_t s=sizeof(float);
	//device
	float *d_a;
	float *d_b;
	float *d_c;
	
	
	//initialzie host value
	for(int i=0;i<N;i++){
		h_a[i]=float(i);,
		h_b[i]=float(i*3);
	}
	
	cudaMemcpy(d_a,h_a,s,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,s,cudaMemcpyHostToDevice);
	cudaMemcpy(d,c,h_c,s,cudaMemcpyHostToDevice);
	
	int ThreadPerBlock =256;
	int BlockPerGrid=N*(ThreadPerBlock-1)/ThreadPerBlock;
	Add<<<ThreadPerBlock,BlockPerGrid,N)>>>;
	
	cudaMemcpy(h_c,d_c,s,cudaMemcpyDevcieToHost);
	cout<<h_c<<endl;
	
	return 0;
}
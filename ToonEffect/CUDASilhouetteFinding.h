#ifndef CUDA_SILHOUETTE_FINDING_H_
#define CUDA_SILHOUETTE_FINDING_H_

#include <cuda.h>
#include "StdHeader.h"

const int g_BLOCK_SIZE = 256;

struct MeshVertex;

bool cudaInitialization(int indiceNum, int vertexNum);

bool cudaProjInit( int silNum );

bool cudaCullInit( int silNum );

bool cudaRunKernel( int indiceNum );

bool cudaRunProjKernel( int silNum );

bool cudaRunCullKernel(int silNum, int indiceNum);

bool cudaPassDataToGPU( MeshVertex* _meshVertex, WORD* _indices, DWORD* _adjBuffer, 
						D3DXMATRIX* h_matrixWorldView, D3DXMATRIX* h_matrixWorldPrj, 
						int h_maxIndiceNum, int h_maxVertexNum );

bool cudaPassProjVerticesDataToGPU( D3DXVECTOR3* h_edgeVertices, int h_silNum );

bool cudaPassCullDataToGPU( D3DXVECTOR3* h_meshVertexProj, int h_silNum );

bool cudaGetDataFromGPU(bool* h_isSilhouette, int silSize);
bool cudaGetCulledDataFromGPU(bool* h_isSilhouette, int h_silNum);
bool cudaGetProjDataFromGPU(D3DXVECTOR3* h_meshProjVertices, int silSize);

#endif

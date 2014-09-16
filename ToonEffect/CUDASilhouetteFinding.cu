//////////////////////////////////////////////////////////////////////////////////////////////////
// 
// File: CUDASilhouetteFinding.cu
// 
// Author: Ren Yifei, yfren@cs.hku.hk
//
// Desc: CUDA kernel code for GP-GPU processing in parallel
//
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "CUDASilhouetteFinding.h"
#include "CUDADataStructure.h"

int	h_curMaxIndiceNum = 0;
int h_curMaxVertexNum = 0;
int h_curMaxSilNum = 0;

__device__ MeshVertex* 	d_meshVertex = NULL;
__device__ WORD*		d_indices = NULL;
__device__ DWORD*		d_adjBuffer = NULL;
__device__ int*			d_maxIndiceNum = NULL;
__device__ int*			d_silNum = NULL;

__device__ D3DXMATRIX*		d_matrixWorldView  = NULL;
__device__ D3DXMATRIX*		d_matrixProj = NULL;

__device__ D3DXVECTOR3*		d_candidateSilhouetteVertex = NULL;

//复用两次，第一次表示是否sil,大小indiceNum/2,第二次就表示是否可见的sil, 大小只用了前面的silNum个
__device__ bool*			d_isSilhouette  = NULL; 

//Silhouette detection
__global__ void findSilhouette(MeshVertex* d_meshVertex,
							   WORD* d_indices, 
							   DWORD* d_adjBuffer, 
							   D3DXMATRIX* d_matrixWorldView,
							   D3DXMATRIX* d_matrixProj,
							   bool*	d_isSilhouette,
							   int* d_maxIndiceNum);

//Invisible silhouette culling
__global__ void cullSilouette(MeshVertex* d_meshVertex,
							 WORD* d_indices,
							 int* d_maxIndiceNum,
							 D3DXVECTOR3* d_candidateSilhouetteVertex,
							 int* d_silNum,
							 bool*	d_isSilhouette,
							 D3DXMATRIX* d_matrixWorldView);


//Projection transform from 3D tO 2D viewport
__global__ void projTransform(D3DXVECTOR3*  d_meshVertexProj,
							  int*			d_silNum,
							  D3DXMATRIX*	d_matrixWorldView,
							  D3DXMATRIX*	d_matrixProj);

//Segment / Triangle crossing testing
__device__ bool segmentIntersectTriangle(const D3DXVECTOR3& orig, 
										 const D3DXVECTOR3& des,
										 const D3DXVECTOR3& v0, 
										 const D3DXVECTOR3& v1, 
										 const D3DXVECTOR3& v2);


//Init
bool cudaInitialization(int indiceNum, int vertexNum)
{	
	cudaError err = cudaSuccess;

	if(vertexNum > h_curMaxVertexNum)
	{
		h_curMaxVertexNum = vertexNum;

		if(d_meshVertex)
			cudaFree(d_meshVertex);
		
		err = cudaMalloc((void**)&d_meshVertex, vertexNum * sizeof(MeshVertex));

		if(err != cudaSuccess)
			return false;
	}
	
	if(indiceNum > h_curMaxIndiceNum)
	{
		h_curMaxIndiceNum = indiceNum;

		if(d_indices)
			cudaFree(d_indices);
		
		err = cudaMalloc((void**)&d_indices, indiceNum * sizeof(WORD));

		if(err != cudaSuccess)
			return false;

		if(d_adjBuffer)
			cudaFree(d_adjBuffer);

		err = cudaMalloc((void**)&d_adjBuffer, indiceNum * sizeof(DWORD));

		if(err != cudaSuccess)
			return false;

		if(d_maxIndiceNum)
			cudaFree(d_maxIndiceNum);

		err = cudaMalloc((void**)&d_maxIndiceNum,		sizeof(int));

		if(err != cudaSuccess)
			return false;

		if(d_matrixWorldView)
			cudaFree(d_matrixWorldView);

		err = cudaMalloc((void**)&d_matrixWorldView,	sizeof(D3DXMATRIX));

		if(err != cudaSuccess)
			return false;

		if(d_matrixProj)
			cudaFree(d_matrixProj);

		err = cudaMalloc((void**)&d_matrixProj,	sizeof(D3DXMATRIX));

		if(err != cudaSuccess)
			return false;

		if(d_isSilhouette)
			cudaFree(d_isSilhouette);

		err = cudaMalloc((void**)&d_isSilhouette, indiceNum * sizeof(bool));

		if(err != cudaSuccess)
			return false;
	}

	return true;
}

bool cudaProjInit( int silNum )
{
	cudaError err = cudaSuccess;

	if(silNum > h_curMaxSilNum)
	{
		h_curMaxSilNum = silNum;

		if(d_candidateSilhouetteVertex)
			cudaFree(d_candidateSilhouetteVertex);

		err = cudaMalloc((void**)&d_candidateSilhouetteVertex, silNum * 2 * sizeof(D3DXVECTOR3));

		if(err != cudaSuccess)
			return false;

		if(d_silNum)
			cudaFree(d_silNum);

		err = cudaMalloc((void**)&d_silNum, sizeof(int));

		if(err != cudaSuccess)
			return false;
	}

	return true;
}



bool cudaPassDataToGPU( MeshVertex* _meshVertex, WORD* _indices, DWORD* _adjBuffer, 
						D3DXMATRIX* h_matrixWorldView, D3DXMATRIX* h_matrixProj, 
						int h_indiceNum, int h_vertexNum )
{
	if(!cudaInitialization(h_indiceNum, h_vertexNum))
		return false;
	
	cudaMemcpy(d_meshVertex, _meshVertex,		h_vertexNum * sizeof(MeshVertex),		cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, _indices,				h_indiceNum * sizeof(WORD),				cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjBuffer, _adjBuffer,			h_indiceNum * sizeof(DWORD),			cudaMemcpyHostToDevice);
	cudaMemcpy(d_maxIndiceNum, &h_indiceNum,					 sizeof(int),			cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixWorldView, h_matrixWorldView,			 sizeof(D3DXMATRIX),	cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixProj, h_matrixProj,						 sizeof(D3DXMATRIX),	cudaMemcpyHostToDevice);

	return true;
}

bool cudaPassProjVerticesDataToGPU( D3DXVECTOR3* edgeVertices, int h_silNum )
{
	if(!cudaProjInit(h_silNum))
		return false;

	cudaMemcpy(d_silNum, &h_silNum, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_candidateSilhouetteVertex, edgeVertices, h_silNum * 2 * sizeof(D3DXVECTOR3), cudaMemcpyHostToDevice);

	return true;
}

bool cudaGetDataFromGPU( bool* h_isSilhouette, int silSize )
{
	cudaMemcpy(h_isSilhouette, d_isSilhouette,	 silSize * sizeof(bool), cudaMemcpyDeviceToHost);
	
	return true;
}

bool cudaGetProjDataFromGPU( D3DXVECTOR3* h_meshProjVertices, int silSize )
{
	cudaMemcpy(h_meshProjVertices, d_candidateSilhouetteVertex, silSize * 2 * sizeof(D3DXVECTOR3), cudaMemcpyDeviceToHost);

	return true;
}

bool cudaRunKernel(int indiceNum)
{
	int gridNum = (indiceNum / g_BLOCK_SIZE);
	
	if(indiceNum % g_BLOCK_SIZE != 0)
		++gridNum;


	findSilhouette<<< gridNum, g_BLOCK_SIZE>>> (d_meshVertex, d_indices, d_adjBuffer, 
												d_matrixWorldView, d_matrixProj,
												d_isSilhouette, d_maxIndiceNum);

	cudaThreadSynchronize();

	return true;
}


bool cudaRunProjKernel(int silNum)
{
	int gridNum = (silNum * 2 / g_BLOCK_SIZE);

	if( (silNum * 2) % g_BLOCK_SIZE != 0 )
		++gridNum;

	projTransform<<< gridNum, g_BLOCK_SIZE>>> (d_candidateSilhouetteVertex, d_silNum, d_matrixWorldView, d_matrixProj);

	cudaThreadSynchronize();

	return true;
}

bool cudaRunCullKernel(int silNum, int indiceNum)
{
	int maxTriangleNum = indiceNum / 3;
	
	int gridNum = (silNum * maxTriangleNum / g_BLOCK_SIZE);

	if( (silNum * maxTriangleNum) % g_BLOCK_SIZE != 0 )
		++gridNum;

	cudaMemset(d_isSilhouette, 1, sizeof(bool)*silNum);

	cullSilouette<<< gridNum, g_BLOCK_SIZE>>> (d_meshVertex, d_indices, d_maxIndiceNum, 
											  d_candidateSilhouetteVertex, d_silNum, d_isSilhouette, 
											  d_matrixWorldView);
	cudaThreadSynchronize();

	return true;
}


__device__  D3DXVECTOR3 crossProduct(const D3DXVECTOR3& m1, const D3DXVECTOR3& m2)
{
	D3DXVECTOR3 ret;

	ret.x = m1.y * m2.z - m1.z * m2.y;
	ret.y = m1.z * m2.x - m1.x * m2.z;
	ret.z = m1.x * m2.y - m1.y * m2.x;

	return ret;
}

__device__ float dotProduct(const D3DXVECTOR3& m1, const D3DXVECTOR3& m2)
{
	float ret = m1.x * m2.x + m1.y * m2.y + m1.z * m2.z;

	return ret;
}

__device__ float length(const D3DXVECTOR3& vec)
{
	return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__device__ D3DXVECTOR3 normalize(const D3DXVECTOR3& vec)
{
	D3DXVECTOR3 ret = vec;

	float len = length(ret);

	ret.x /= len;
	ret.y /= len;
	ret.z /= len;

	return ret;
}

__device__ D3DXVECTOR3 matrixPntMul(const D3DXVECTOR3& pnt, const D3DXMATRIX* mat)
{
	
	D3DXVECTOR3 ret;
	
	ret.x = mat->m[0][0] * pnt.x + mat->m[1][0] * pnt.y + mat->m[2][0] * pnt.z + mat->m[3][0];
	ret.y = mat->m[0][1] * pnt.x + mat->m[1][1] * pnt.y + mat->m[2][1] * pnt.z + mat->m[3][1];
	ret.z = mat->m[0][2] * pnt.x + mat->m[1][2] * pnt.y + mat->m[2][2] * pnt.z + mat->m[3][2];
	
	float w = mat->m[0][3] * pnt.x + mat->m[1][3] * pnt.y + mat->m[2][3] * pnt.z + mat->m[3][3];

	ret.x /= w;
	ret.y /= w;
	ret.z /= w;

	return ret;
}

__device__ D3DXVECTOR3 matrixVecMul(const D3DXVECTOR3& vec, const D3DXMATRIX* mat)
{

	D3DXVECTOR3 ret;

	ret.x = mat->m[0][0] * vec.x + mat->m[1][0] * vec.y + mat->m[2][0] * vec.z;
	ret.y = mat->m[0][1] * vec.x + mat->m[1][1] * vec.y + mat->m[2][1] * vec.z;
	ret.z = mat->m[0][2] * vec.x + mat->m[1][2] * vec.y + mat->m[2][2] * vec.z;

	return ret;
}


__global__ void findSilhouette(MeshVertex* d_meshVertex,
							   WORD* d_indices, 
							   DWORD* d_adjBuffer, 
							   D3DXMATRIX* d_matrixWorldView,
							   D3DXMATRIX* d_matrixProj,
							   bool*	d_isSilhouette,
							   int* d_maxIndiceNum)
{
	const int idx = blockIdx.x * g_BLOCK_SIZE + threadIdx.x;

	if(idx >= *d_maxIndiceNum)
		return;
	
	const int idxTriangle	  = idx / 3;
	const int idxTriangleBase = idxTriangle * 3;

	const int idxV0				= d_indices[idxTriangleBase];
	const int idxV1				= d_indices[idxTriangleBase + 1];
	const int idxV2				= d_indices[idxTriangleBase + 2];

	const D3DXVECTOR3& posV0	= d_meshVertex[idxV0].position;
	const D3DXVECTOR3& posV1	= d_meshVertex[idxV1].position;
	const D3DXVECTOR3& posV2	= d_meshVertex[idxV2].position;

	const D3DXVECTOR3 vecV0V1	= posV1 - posV0;
	const D3DXVECTOR3 vecV0V2	= posV2 - posV0;

	D3DXVECTOR3 normal1	= crossProduct(vecV0V1, vecV0V2);

	D3DXVECTOR3 normal2;
	const int idxAdjTriangle = d_adjBuffer[idx];
	
	if(idxAdjTriangle != -1)
	{
		const int idxAdjTriangleBase = idxAdjTriangle * 3;

		const int idxAdjV0			= d_indices[idxAdjTriangleBase];
		const int idxAdjV1			= d_indices[idxAdjTriangleBase + 1];
		const int idxAdjV2			= d_indices[idxAdjTriangleBase + 2];

		const D3DXVECTOR3& posAdjV0	= d_meshVertex[idxAdjV0].position;
		const D3DXVECTOR3& posAdjV1	= d_meshVertex[idxAdjV1].position;
		const D3DXVECTOR3& posAdjV2	= d_meshVertex[idxAdjV2].position;

		const D3DXVECTOR3 vecAdjV0V1	= posAdjV1 - posAdjV0;
		const D3DXVECTOR3 vecAdjV0V2	= posAdjV2 - posAdjV0;

		normal2 = crossProduct(vecAdjV0V1, vecAdjV0V2);
	}
	else
	{
		normal2 = -normal1;
	}

	D3DXVECTOR3 eyeToVertex = matrixPntMul(posV0, d_matrixWorldView);

	normal1 = matrixVecMul(normal1, d_matrixWorldView);
	normal2 = matrixVecMul(normal2, d_matrixWorldView);

	float dot1 = dotProduct(normal1, eyeToVertex);
	float dot2 = dotProduct(normal2, eyeToVertex);
	
	if(dot1 * dot2 < 0.0f)
	{
		//It's a silhouette
		d_isSilhouette[idx] = true;
	}
	else
	{
		d_isSilhouette[idx] = false;
	}
}

__global__ void projTransform( D3DXVECTOR3* d_meshVertexProj,
							   int*			d_silNum,
							   D3DXMATRIX*	d_matrixWorldView,
							   D3DXMATRIX*	d_matrixProj)
{
	const int idx = blockIdx.x * g_BLOCK_SIZE + threadIdx.x;

	int silVerticesNum = (*d_silNum) * 2;
 
	if(idx >= silVerticesNum)
		return;

	//Projection Transformation
	d_meshVertexProj[idx] = matrixPntMul(d_meshVertexProj[idx], d_matrixWorldView);
	d_meshVertexProj[idx] = matrixPntMul(d_meshVertexProj[idx], d_matrixProj);
}

__global__ void cullSilouette(MeshVertex* d_meshVertex,
							 WORD* d_indices,
							 int* d_maxIndiceNum,
							 D3DXVECTOR3* d_candidateSilhouetteVertex,
							 int*	d_silNum,
							 bool*	d_isSilhouette,
							 D3DXMATRIX* d_matrixWorldView)
{
	const int idx = blockIdx.x * g_BLOCK_SIZE + threadIdx.x;

	int triangleNum = *d_maxIndiceNum / 3;

	if(idx >= *d_silNum * triangleNum)
		return;

	int silIdx = idx / triangleNum;

	if(!d_isSilhouette[silIdx])
		return;

	int triangleIdx = idx % triangleNum;

	D3DXVECTOR3 endPnt1 = matrixPntMul(d_candidateSilhouetteVertex[2*silIdx], d_matrixWorldView);
	D3DXVECTOR3 endPnt2 = matrixPntMul(d_candidateSilhouetteVertex[2*silIdx+1], d_matrixWorldView);

	D3DXVECTOR3 silMidPnt = (endPnt1 + endPnt2) / 2.0f;

	WORD triangleV0Idx = d_indices[3*triangleIdx];
	WORD triangleV1Idx = d_indices[3*triangleIdx+1];
	WORD triangleV2Idx = d_indices[3*triangleIdx+2];

	D3DXVECTOR3 v0Pos = d_meshVertex[triangleV0Idx].position;
	D3DXVECTOR3 v1Pos = d_meshVertex[triangleV1Idx].position;
	D3DXVECTOR3 v2Pos = d_meshVertex[triangleV2Idx].position;

	v0Pos = matrixPntMul(v0Pos, d_matrixWorldView);
	v1Pos = matrixPntMul(v1Pos, d_matrixWorldView);
	v2Pos = matrixPntMul(v2Pos, d_matrixWorldView);

	D3DXVECTOR3 origin = D3DXVECTOR3(0,0,0);
	bool isInvisible = segmentIntersectTriangle(origin, silMidPnt, v0Pos, v1Pos, v2Pos);

	if(isInvisible)
	{
		d_isSilhouette[silIdx] = false;
	}
}

__device__ bool segmentIntersectTriangle(const D3DXVECTOR3& orig, 
										 const D3DXVECTOR3& des,
										 const D3DXVECTOR3& v0, 
										 const D3DXVECTOR3& v1, 
										 const D3DXVECTOR3& v2)
{
	float t,u,v;
	const D3DXVECTOR3 tmpDir = des - orig;
	
	D3DXVECTOR3 dir = normalize(tmpDir);

	// Find vectors for two edges sharing vert0
	D3DXVECTOR3 edge1 = v1 - v0;
	D3DXVECTOR3 edge2 = v2 - v0;

	// Begin calculating determinant - also used to calculate U parameter
	D3DXVECTOR3 pvec;
	pvec = crossProduct(dir, edge2);

	// If determinant is near zero, ray lies in plane of triangle
	float det = dotProduct(edge1, pvec);

	D3DXVECTOR3 tvec;
	if( det > 0 )
	{
		tvec = orig - v0;
	}
	else
	{
		tvec = v0 - orig;
		det = -det;
	}

	if( det < 0.0001f )
		return false;

	// Calculate U parameter and test bounds
	u = dotProduct(tvec, pvec);

	if( u < 0.0f || u > det )
		return false;

	// Prepare to test V parameter
	D3DXVECTOR3 qvec;
	qvec = crossProduct(tvec, edge1);

	// Calculate V parameter and test bounds
	v = dotProduct(dir, qvec);

	if( v < 0.0f || u + v > det )
		return false;

	// Calculate t, scale parameters, ray intersects triangle
	t = dotProduct(edge2, qvec);
	FLOAT fInvDet = 1.0f / det;
	t *= fInvDet;
	u *= fInvDet;
	v *= fInvDet;
	
	if( length(orig + t * dir) > length(des - orig) )
		return false;
	else if( fabs(length(orig + t * dir) - length(des - orig)) < 0.0001 )
		return false;
		
	return true;
}

bool cudaGetCulledDataFromGPU( bool* h_isSilhouette, int h_silNum )
{
	cudaMemcpy(h_isSilhouette, d_isSilhouette,	 h_silNum * sizeof(bool), cudaMemcpyDeviceToHost);

	return true;
}

bool cudaCullInit( int silNum )
{
	cudaError err = cudaSuccess;

	if(silNum > h_curMaxSilNum)
	{
		h_curMaxSilNum = silNum;

		if(d_candidateSilhouetteVertex)
			cudaFree(d_candidateSilhouetteVertex);

		err = cudaMalloc((void**)&d_candidateSilhouetteVertex, silNum * 2 * sizeof(D3DXVECTOR3));

		if(err != cudaSuccess)
			return false;

		if(d_silNum)
			cudaFree(d_silNum);

		err = cudaMalloc((void**)&d_silNum, sizeof(int));

		if(err != cudaSuccess)
			return false;
	}

	return true;
}

bool cudaPassCullDataToGPU( D3DXVECTOR3* h_meshVertexProj, int h_silNum )
{
	if(!cudaCullInit(h_silNum))
		return false;

	cudaMemcpy(d_silNum, &h_silNum, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_candidateSilhouetteVertex, h_meshVertexProj, h_silNum * 2 * sizeof(D3DXVECTOR3), cudaMemcpyHostToDevice);

	return true;
}
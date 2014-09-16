#ifndef CUDA_DATA_STRUCTURE_H_
#define CUDA_DATA_STRUCTURE_H_

#include "StdHeader.h"

struct EdgeVertex
{
	D3DXVECTOR3 position;
	D3DXVECTOR3 normal;
	D3DXVECTOR3 silhouetteWidth;
	D3DXVECTOR3	silhouetteAlpha;

	D3DXVECTOR2 texCoord;
};


struct MeshVertex // 24 BYTEs
{
	D3DXVECTOR3 position;
	D3DXVECTOR3 normal;
};

struct SegmentGroup
{
	int groupIdx;
	int offsetIdx;
};

struct SegmentGroupInfo
{
	int total;
	int minIdx;
};

#endif
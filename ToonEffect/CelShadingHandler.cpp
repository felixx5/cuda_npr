//////////////////////////////////////////////////////////////////////////////////////////////////
// 
// File: CelShadingHandler.cpp
// 
// Author: Ren Yifei, yfren@cs.hku.hk
//
// Desc: The manager for silhouette processing
//
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelShadingHandler.h"
#include "CUDADataStructure.h"
#include "CUDASilhouetteFinding.h"
#include "CelSilhouette.h"
#include "d3dUtility.h"

extern bool g_randomWiggling;
extern bool g_alphaTransition;
extern bool g_widthTransition;

float CelShadingHandler::s_ConnectDisThreshold = 0.03f;
float CelShadingHandler::s_ConnectAngleThreshold = .90f;

CelShadingHandler::CelShadingHandler(IDirect3DDevice9* device) : 
m_isSilhouette(NULL),
m_candidateSilhouetteVertex(NULL),
m_candidateSilhouetteVertexNormal(NULL),
m_segGroup(NULL),
m_segGroupInfo(NULL),
m_isSilhouetteSize(0),
m_candidateSilhouetteVertexNum(0),
m_indicesNum(0),
m_vertexNum(0),
m_silNum(0)
{

}

CelShadingHandler::~CelShadingHandler()
{
	delete [] m_isSilhouette;
	delete [] m_candidateSilhouetteVertex;
	delete [] m_candidateSilhouetteVertexNormal;
	delete [] m_segGroup;
	delete [] m_segGroupInfo;
}


bool CelShadingHandler::passDataToGPU(MeshVertex* h_meshVertex, 
									  WORD* h_indices, 
									  DWORD* h_adjBuffer, 
									  D3DXMATRIX* h_matrixWorldView,
									  D3DXMATRIX* h_matrixWorldProj,
									  int h_indicesNum,
									  int h_vertexNum)
{
	return cudaPassDataToGPU(h_meshVertex, h_indices, h_adjBuffer, h_matrixWorldView, h_matrixWorldProj, h_indicesNum, h_vertexNum);
}

bool CelShadingHandler::getDataFromGPU()
{
	if(m_indicesNum > m_isSilhouetteSize)
	{
		if(m_isSilhouette)
		{
			delete [] m_isSilhouette;
		}

		m_isSilhouette = new bool[m_indicesNum];
	}
	m_isSilhouetteSize = m_indicesNum;

	return cudaGetDataFromGPU(m_isSilhouette, m_isSilhouetteSize);
}

bool CelShadingHandler::runKernel(int indiceNum)
{
	return cudaRunKernel(indiceNum);	
}

bool CelShadingHandler::process(CelSilhouette* celSilhouette, D3DXMATRIX* worldViewMat, D3DXMATRIX* projMat)
{
	if(!celSilhouette)
		return false;

	int indicesNum = celSilhouette->m_indicesNum;
	int vertexNum = celSilhouette->m_vertexNum;

	m_indicesNum = indicesNum;
	m_vertexNum = vertexNum;

	ID3DXMesh* mesh = celSilhouette->m_mesh;

	WORD* celIndices = 0;
	mesh->LockIndexBuffer(0, (void**)&celIndices);

	MeshVertex* meshVertices = 0;
	mesh->LockVertexBuffer(0, (void**)&meshVertices);

	DWORD* adjPointer = (DWORD*)celSilhouette->m_adjBuffer->GetBufferPointer();
	
	this->passDataToGPU(meshVertices, celIndices, adjPointer, worldViewMat, projMat, m_indicesNum, m_vertexNum);

	this->runKernel(m_indicesNum);

	this->getDataFromGPU();

	this->generateQuads(celSilhouette,meshVertices, celIndices);

	mesh->UnlockVertexBuffer();
	mesh->UnlockIndexBuffer();
	
	return true;
}

bool CelShadingHandler::generateQuads(CelSilhouette* celSihouette, MeshVertex* meshVertices, WORD* celIndices)
{	
	if ( !this->generateSilhouetteCandidates(meshVertices, celIndices))
		return false;

	celSihouette->createBuffer(m_silNum);
	WORD* edgeIndices = 0;
	celSihouette->m_ib->Lock(0, 0, (void**)&edgeIndices, 0);

	EdgeVertex* edgeVertices = 0;
	celSihouette->m_vb->Lock(0, 0, (void**)&edgeVertices, 0);
	EdgeVertex* edgeVerticesHead = edgeVertices;

	if( !this->generateSilhouettes(celSihouette, celIndices, edgeVerticesHead, edgeIndices))
		return false;

	if( !this->connectSegments(edgeVerticesHead) )
		return false;
	
	celSihouette->m_ib->Unlock();
	celSihouette->m_vb->Unlock();
	
	return true;
}

float CelShadingHandler::generateWeight() const
{
	int randFactor = rand() % 100;

	return float(randFactor / 100.0);
}

bool CelShadingHandler::connectSegments(EdgeVertex* edgeVerticesHead)
{
	if( !this->vertexProjTransform(edgeVerticesHead))
		return false;

	memset(m_segGroup,		0, sizeof(SegmentGroup) * m_silNum);
	memset(m_segGroupInfo,	0, sizeof(SegmentGroupInfo) * (m_silNum+1));
	
	int segGroupId = 1;

	for(int i=0; i<m_silNum; ++i)
	{
		if(m_segGroup[i].groupIdx == 0)
		{
			m_segGroup[i].groupIdx = segGroupId;
			++m_segGroupInfo[segGroupId].total;
			
			dfs(2*i, 2*i+1);
			
			++segGroupId;
		}
	}

	this->adjustSilouette(edgeVerticesHead);

	return true;
}

bool CelShadingHandler::connectivityTest( const D3DXVECTOR3& a, const D3DXVECTOR3& b, float& dis )
{
	float disSquare= (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);// + (a.z - b.z)*(a.z - b.z);

 	if(disSquare > s_ConnectDisThreshold)
 		return false;

	dis = disSquare;

// 	float crossAngle = (a.x * b.x + a.y * b.y) / (sqrt(a.x * a.x + a.y * a.y) * sqrt(b.x * b.x + b.y * b.y));
// 	
// 	if(crossAngle < s_ConnectAngleThreshold)
// 		return false;
	
	return true;
}

void CelShadingHandler::adjustSilouette( EdgeVertex* edgeVerticesHead )
{
	for(int i=0; i<m_silNum; ++i)
	{
		int groupId  = m_segGroup[i].groupIdx;
		int myOffset = m_segGroup[i].offsetIdx;
		
		if(groupId==0)
			continue;
		
		float baseWidth = 0.0;
		float headPos = float(myOffset - m_segGroupInfo[groupId].minIdx) / m_segGroupInfo[groupId].total;
		float intervalLength = 1.0f / m_segGroupInfo[groupId].total;
		
		if(g_widthTransition)
		{
			edgeVerticesHead[4*i+0].silhouetteWidth.z *= (headPos + baseWidth);
			edgeVerticesHead[4*i+1].silhouetteWidth.z *= (headPos + baseWidth);
			edgeVerticesHead[4*i+2].silhouetteWidth.z *= (headPos + baseWidth);
			edgeVerticesHead[4*i+3].silhouetteWidth.z *= (headPos + baseWidth);
		}

		//Set alpha
		if(g_alphaTransition)
		{
			edgeVerticesHead[4*i+0].silhouetteAlpha.x = (headPos + baseWidth);
			edgeVerticesHead[4*i+1].silhouetteAlpha.x = (headPos + baseWidth);
			edgeVerticesHead[4*i+2].silhouetteAlpha.x = (headPos + baseWidth);
			edgeVerticesHead[4*i+3].silhouetteAlpha.x = (headPos + baseWidth);
		}
		else
		{
			edgeVerticesHead[4*i+0].silhouetteAlpha.x = 0;
			edgeVerticesHead[4*i+1].silhouetteAlpha.x = 0;
			edgeVerticesHead[4*i+2].silhouetteAlpha.x = 0;
			edgeVerticesHead[4*i+3].silhouetteAlpha.x = 0;
		}
		
		if(g_randomWiggling)
		{
			this->setSegmentRandomBias(edgeVerticesHead, 4*i+0);
			this->setSegmentRandomBias(edgeVerticesHead, 4*i+1);
			this->setSegmentRandomBias(edgeVerticesHead, 4*i+2);
			this->setSegmentRandomBias(edgeVerticesHead, 4*i+3);
		}

		edgeVerticesHead[4*i+0].texCoord.x = headPos;
		edgeVerticesHead[4*i+0].texCoord.y = 0.0f;

		edgeVerticesHead[4*i+1].texCoord.x = headPos + intervalLength;
		edgeVerticesHead[4*i+1].texCoord.y = 0.0f;

		edgeVerticesHead[4*i+2].texCoord.x = headPos;
		edgeVerticesHead[4*i+2].texCoord.y = 1.0f;
		
		edgeVerticesHead[4*i+3].texCoord.x = headPos + intervalLength;
		edgeVerticesHead[4*i+3].texCoord.y = 1.0f;
	}
}

void CelShadingHandler::dfs( int leftEndPntIdx, int rightEndPntIdx )
{
	while(true)
	{
		float minDis		= 1000000.0f;
		float curDis		= 0;
		int   minIdx		= -1;
		int	  expiredSide	= 0;

		const D3DXVECTOR3& leftEndPnt	= m_candidateSilhouetteVertex[leftEndPntIdx];
		const D3DXVECTOR3& rightEndPnt	= m_candidateSilhouetteVertex[rightEndPntIdx];

		for(int i=0; i<m_silNum; ++i)
		{
			if(m_segGroup[i].groupIdx!=0)
				continue;

			if(this->connectivityTest(m_candidateSilhouetteVertex[2*i], leftEndPnt, curDis))
			{
				if(curDis < minDis)
				{
					minDis = curDis;
					minIdx = 2*i+1; //my brother connects with someone else, so it's me who should substitutes that guy.
					expiredSide = 1;
				}
			}
			else if(this->connectivityTest(m_candidateSilhouetteVertex[2*i+1], leftEndPnt, curDis))
			{
				if(curDis < minDis)
				{
					minDis = curDis;
					minIdx = 2*i;
					expiredSide = 1;
				}
			}
			else if(this->connectivityTest(m_candidateSilhouetteVertex[2*i+1], rightEndPnt, curDis))
			{
				if(curDis < minDis)
				{
					minDis = curDis;
					minIdx = 2*i;
					expiredSide = 2;
				}
			}
			else if(this->connectivityTest(m_candidateSilhouetteVertex[2*i], rightEndPnt, curDis))
			{
				if(curDis < minDis)
				{
					minDis = curDis;
					minIdx = 2*i+1;
					expiredSide = 2;
				}
			}
		}

		//found a connective point
		if(minIdx != -1)
		{
			//set group ID
			int curIdx = minIdx / 2;

			int newLeftEndPntIdx(leftEndPntIdx);
			int newRightEndPntIdx(rightEndPntIdx);

			if(expiredSide == 1)
			{
				newLeftEndPntIdx = minIdx;

				int groupIdx = m_segGroup[leftEndPntIdx / 2].groupIdx;
				m_segGroup[curIdx].groupIdx = groupIdx;

				int offsetIdx = m_segGroup[leftEndPntIdx / 2].offsetIdx - 1;
				m_segGroup[curIdx].offsetIdx = offsetIdx;

				m_segGroupInfo[groupIdx].minIdx = offsetIdx;
				++(m_segGroupInfo[groupIdx].total);
			}
			else if(expiredSide == 2)
			{
				newRightEndPntIdx = minIdx;

				int groupIdx = m_segGroup[rightEndPntIdx / 2].groupIdx;
				m_segGroup[curIdx].groupIdx = m_segGroup[rightEndPntIdx / 2].groupIdx;

				int offsetIdx = m_segGroup[rightEndPntIdx / 2].offsetIdx + 1;
				m_segGroup[curIdx].offsetIdx = offsetIdx;

				++(m_segGroupInfo[groupIdx].total);
			}

			leftEndPntIdx  = newLeftEndPntIdx;
			rightEndPntIdx = newRightEndPntIdx;
		}
		else
			break;
	}
}                                                                                                                                                             

bool CelShadingHandler::vertexProjTransform( EdgeVertex* edgeVertices )
{
	for(int i=0; i<m_silNum; ++i)
	{
		m_candidateSilhouetteVertex[2*i] = edgeVertices[4*i].position;
		m_candidateSilhouetteVertex[2*i+1] = edgeVertices[4*i+1].position;
	}

	if( !cudaPassProjVerticesDataToGPU(m_candidateSilhouetteVertex, m_silNum))
		return false;

	cudaRunProjKernel(m_silNum);

	cudaGetProjDataFromGPU(m_candidateSilhouetteVertex, m_silNum);

	return true;
}

void CelShadingHandler::calPerpendicularUnitVector(EdgeVertex* edgeVerticesHead)
{
	//Calculate the 2D vector is perpendicular to current silhouette after projection.
	for(int i=0; i<m_silNum; ++i)
	{
		for(int j=0; j<4; ++j)
		{
			edgeVerticesHead[4*i+j].silhouetteWidth.x = m_candidateSilhouetteVertex[2*i+1].y - m_candidateSilhouetteVertex[2*i].y;
			edgeVerticesHead[4*i+j].silhouetteWidth.y = m_candidateSilhouetteVertex[2*i].x - m_candidateSilhouetteVertex[2*i+1].x;
		}
	}
}

bool CelShadingHandler::initMeshVertexBuffer()
{
	int silVerticesNum = m_silNum * 2;

	if(silVerticesNum > m_candidateSilhouetteVertexNum)
	{
		if(m_candidateSilhouetteVertex)
		{
			delete [] m_candidateSilhouetteVertex;
		}

		m_candidateSilhouetteVertex = new D3DXVECTOR3[silVerticesNum];

		if(m_candidateSilhouetteVertexNormal)
		{
			delete [] m_candidateSilhouetteVertexNormal;
		}

		m_candidateSilhouetteVertexNormal = new D3DXVECTOR3[silVerticesNum];

		if(m_segGroup)
		{
			delete [] m_segGroup;
		}

		m_segGroup = new SegmentGroup[m_silNum];

		if(m_segGroupInfo)
		{
			delete [] m_segGroupInfo;
		}

		m_segGroupInfo = new SegmentGroupInfo[m_silNum +1];
	}

	m_candidateSilhouetteVertexNum = silVerticesNum;

	return true;
}

bool CelShadingHandler::cullInvisibleSilouette()
{
	if( !cudaPassCullDataToGPU(m_candidateSilhouetteVertex, m_silNum))
		return false;

	cudaRunCullKernel(m_silNum, m_indicesNum);

	cudaGetCulledDataFromGPU(m_isSilhouette, m_silNum);

	return true;
}

bool CelShadingHandler::generateSilhouetteCandidates( MeshVertex* meshVertices, WORD* celIndices )
{
	m_silNum = 0;
	
	for(int i=0; i<m_isSilhouetteSize; ++i)
	{
		if(m_isSilhouette[i])
			++m_silNum;
	}

	if( !this->initMeshVertexBuffer() )
		return false;

	int silCandidateIdx = 0;

	for(int i=0; i<m_isSilhouetteSize; ++i)
	{
		if(m_isSilhouette[i])
		{
			int idxTriangle = i / 3;
			int idxMod = i % 3;

			int idxStart	= celIndices[3 * idxTriangle + idxMod];
			int idxEnd		= celIndices[3 * idxTriangle + (idxMod+1)%3];

			m_candidateSilhouetteVertex[silCandidateIdx] = meshVertices[idxStart].position;
			m_candidateSilhouetteVertexNormal[silCandidateIdx] = meshVertices[idxStart].normal;
			++silCandidateIdx;

			m_candidateSilhouetteVertex[silCandidateIdx] = meshVertices[idxEnd].position;
			m_candidateSilhouetteVertexNormal[silCandidateIdx] = meshVertices[idxEnd].normal;
			++silCandidateIdx;
		}
	}

	if( !this->cullInvisibleSilouette() )
		return false;

	return true;
}

bool CelShadingHandler::generateSilhouettes( CelSilhouette* celSihouette, WORD* celIndices, EdgeVertex* edgeVertices, WORD* edgeIndices )
{
	//Recaculate the silhouette num after culling.
	int realSilNum = 0;

	for(int i=0; i<m_silNum; ++i)
	{
		if(m_isSilhouette[i])
		{
			++realSilNum;

			edgeVertices->position = m_candidateSilhouetteVertex[2*i];
			edgeVertices->normal = m_candidateSilhouetteVertexNormal[2*i];
			edgeVertices->silhouetteWidth.z = -1;
			++edgeVertices;

			edgeVertices->position = m_candidateSilhouetteVertex[2*i+1];
			edgeVertices->normal = m_candidateSilhouetteVertexNormal[2*i+1];
			edgeVertices->silhouetteWidth.z = -1;
			++edgeVertices;

			edgeVertices->position = m_candidateSilhouetteVertex[2*i];
			edgeVertices->normal = m_candidateSilhouetteVertexNormal[2*i];
			edgeVertices->silhouetteWidth.z = 1;
			++edgeVertices;

			edgeVertices->position = m_candidateSilhouetteVertex[2*i+1];
			edgeVertices->normal = m_candidateSilhouetteVertexNormal[2*i+1];
			edgeVertices->silhouetteWidth.z = 1;
			++edgeVertices;
		}
	}

	for(int i = 0; i<realSilNum; ++i)
	{
		edgeIndices[i * 6 + 0] = i * 4;
		edgeIndices[i * 6 + 1] = i * 4 + 1;                           
		edgeIndices[i * 6 + 2] = i * 4 + 2;
		edgeIndices[i * 6 + 3] = i * 4 + 1;
		edgeIndices[i * 6 + 4] = i * 4 + 3;
		edgeIndices[i * 6 + 5] = i * 4 + 2;
	}
	
	m_silNum = realSilNum;
	
	return true;
}

void CelShadingHandler::setSegmentRandomBias( EdgeVertex* edgeVerticesHead, int idx )
{
	int weightX = edgeVerticesHead[idx].normal.x * 10000;
	int weightY = edgeVerticesHead[idx].normal.y * 10000;
	int weightZ = edgeVerticesHead[idx].normal.z * 10000;

	float randomFactor = (abs(weightX + weightY + weightZ) % 1000) / 1000.0f; // just a hash

	randomFactor = max(randomFactor, 0.1f); // not too narrow

	edgeVerticesHead[idx].silhouetteWidth.z += edgeVerticesHead[idx].silhouetteWidth.z * randomFactor;
}
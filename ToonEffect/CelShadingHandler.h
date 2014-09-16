#ifndef CEL_SHADING_HANDLER_H_
#define CEL_SHADING_HANDLER_H_

#include "StdHeader.h"

struct MeshVertex;
struct EdgeVertex;
struct SegmentGroup;
struct SegmentGroupInfo;

class CelSilhouette;

class CelShadingHandler
{
public:
	
	CelShadingHandler(IDirect3DDevice9* device = NULL);
	virtual ~CelShadingHandler();

	bool process(CelSilhouette* celSilhouette, 
				 D3DXMATRIX* worldViewMat, 
				 D3DXMATRIX* projMat);

protected:

	bool	passDataToGPU(	MeshVertex* h_meshVertex, 
							WORD*		h_indices,
							DWORD*		h_adjBuffer, 
							D3DXMATRIX* h_matrixWorldView,
							D3DXMATRIX* h_matrixWorldProj,
							int			h_indicesNum,
							int			h_vertexNum);

	bool	runKernel(int indiceNum);

	bool	getDataFromGPU();

	bool	generateQuads(	CelSilhouette* celSihouette, 
							MeshVertex* edgeVertices, 
							WORD* celIndices);

	bool	generateSilhouetteCandidates(MeshVertex* meshVertices, 
										 WORD* celIndices);

	bool	generateSilhouettes(CelSilhouette* celSihouette, 
								WORD* celIndices, 
								EdgeVertex* edgeVertices, 
								WORD* edgeIndices);

	bool	initMeshVertexBuffer();
	
	bool	vertexProjTransform(EdgeVertex* edgeVertices);

	bool	cullInvisibleSilouette();

	bool	connectSegments(EdgeVertex* edgeVerticesHead);

	void	adjustSilouette(EdgeVertex* edgeVerticesHead);

	float	generateWeight() const;

	bool	connectivityTest(const D3DXVECTOR3& a, const D3DXVECTOR3& b, float& dis);

	void	setSegmentRandomBias(EdgeVertex* edgeVerticesHead, int idx);

	void	calPerpendicularUnitVector(EdgeVertex* edgeVerticesHead);

	void	dfs( int leftEndPntIdx, int rightEndPntIdx );

private:

	static float s_ConnectDisThreshold;
	static float s_ConnectAngleThreshold;

	int		m_indicesNum;
	int		m_vertexNum;
	int		m_silNum;

	bool*	m_isSilhouette;
	int		m_isSilhouetteSize;

	SegmentGroup*		m_segGroup;
	SegmentGroupInfo*	m_segGroupInfo;

	D3DXVECTOR3*		m_candidateSilhouetteVertex;
	int					m_candidateSilhouetteVertexNum;
	D3DXVECTOR3*		m_candidateSilhouetteVertexNormal;
};

#endif

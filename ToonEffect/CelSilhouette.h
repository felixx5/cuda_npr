#ifndef CEL_SILHOUETTE_H_
#define CEL_SILHOUETTE_H_

#include "StdHeader.h"

class CelSilhouette
{
public:

	friend class CelShadingHandler;

	CelSilhouette(IDirect3DDevice9* device = NULL, 
				  ID3DXMesh* d3dMesh = NULL, 
				  ID3DXBuffer* adjBuffer = NULL);

	virtual ~CelSilhouette();

	bool init(ID3DXMesh* d3dMesh = NULL);

	void render();

	void createBuffer(int size);

protected:

	bool createVertexDeclaration();

private:

	int	m_indicesNum;
	int m_vertexNum;
	
	int m_silhouetteNum;

	ID3DXBuffer* m_adjBuffer;

	IDirect3DDevice9*			 m_device;

	ID3DXMesh*					 m_mesh;

	IDirect3DVertexBuffer9*      m_vb;
	IDirect3DIndexBuffer9*       m_ib;
	IDirect3DVertexDeclaration9* m_decl;
};


#endif
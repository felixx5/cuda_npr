//////////////////////////////////////////////////////////////////////////////////////////////////
// 
// File: CelSilhouette.cpp
// 
// Author: Ren Yifei, yfren@cs.hku.hk
//
// Desc: Handling silhouettes information
//
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelSilhouette.h"
#include "CUDADataStructure.h"
#include "d3dUtility.h"

CelSilhouette::CelSilhouette(IDirect3DDevice9* device, 
							 ID3DXMesh* d3dMesh, 
							 ID3DXBuffer* adjBuffer) 
: 
m_device(device), 
m_adjBuffer(adjBuffer), 
m_vb(NULL), 
m_ib(NULL)
{
	this->init(d3dMesh);
}

bool CelSilhouette::init(ID3DXMesh* d3dMesh)
{
	if(d3dMesh)
	{
		m_vertexNum = d3dMesh->GetNumVertices();
		m_indicesNum = d3dMesh->GetNumFaces() * 3;

		m_mesh = d3dMesh;

		return this->createVertexDeclaration();
	}

	return false;
}

CelSilhouette::~CelSilhouette()
{
	d3d::Release<IDirect3DVertexBuffer9*>(m_vb);
	d3d::Release<IDirect3DIndexBuffer9*>(m_ib);
	d3d::Release<IDirect3DVertexDeclaration9*>(m_decl);
	d3d::Release<ID3DXBuffer*>(m_adjBuffer);
}

void CelSilhouette::render()
{
	m_device->SetVertexDeclaration(m_decl);
	m_device->SetStreamSource(0, m_vb, 0, sizeof(EdgeVertex));
	m_device->SetIndices(m_ib);
	
	m_device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, m_silhouetteNum * 4, 0, m_silhouetteNum * 2);// !!!!!!!!!!!!!!!!
}

bool CelSilhouette::createVertexDeclaration()
{
	HRESULT hr = 0;

	D3DVERTEXELEMENT9 decl[] = 
	{
		// offsets in bytes
		{0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0},
		{0, 24, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   1},
		{0, 36, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   2},
		{0, 48, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};

	hr = m_device->CreateVertexDeclaration(decl, &m_decl);

	if(FAILED(hr))
	{
		::MessageBox(0, "CreateVertexDeclaration() - FAILED", 0, 0);
		return false;
	}

	return true;
}

void CelSilhouette::createBuffer(int size)
{
	m_silhouetteNum = size;

	if(m_vb)
		d3d::Release<IDirect3DVertexBuffer9*>(m_vb);
	
	if(m_ib)
		d3d::Release<IDirect3DIndexBuffer9*>(m_ib);

	int edgeSize = 4 * size;
	int indexSize = 6 * size;

	m_device->CreateVertexBuffer(	edgeSize * sizeof(EdgeVertex),
									D3DUSAGE_WRITEONLY,
									0, // using vertex declaration
									D3DPOOL_MANAGED,
									&m_vb,
									0);

	m_device->CreateIndexBuffer(	indexSize * sizeof(WORD), // 2 triangles per edge
									D3DUSAGE_WRITEONLY,
									D3DFMT_INDEX16,
									D3DPOOL_MANAGED,
									&m_ib,
									0);
}
//////////////////////////////////////////////////////////////////////////////////////////////////
// 
// File: Main.cpp
// 
// Author: Ren Yifei, yfren@cs.hku.hk
//
// Desc: The main process including render loop
//
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "StdHeader.h"
#include "d3dUtility.h"
#include "CelShadingHandler.h"
#include "CelSilhouette.h"

// Globals

const int WIDTH  = 800;
const int HEIGHT = 600;

//Constant strings
const char* CONFIG_FILE_NAME = "./config.ini";

const char* HELP_STRING = "Non-Photorealistic Rendering Demo(CUDA Version)\n\n\
						  Author: Ren Yifei\n\
						  Email: yfren@cs.hku.hk\n\n\
						  Control\n\n\
						  Esc: Quit\n\
						  F1: Hide help\n\
						  F2: Non-Photorealistic Rendering On/Off\n\
						  F3: Render color On/Off\n\
						  F4: Render WireFrame On/Off\n\
						  F5: Stroke random wiggling On/Off\n\
						  F6: Stroke alpha fade out On/Off\n\
						  F7: Stroke width fade out On/Off\n\
						  A:  Increase Stroke width\n\
						  Z:  Decrease Stroke width\n\
						  Up/Down/Left/Right: Move camera\n\n\
						  To add/delete objects or Modify textures for strokes,\n\
						  please modify the config.ini file according to the \n\
						  example format inside it.";

const char* HIDE_STRING = "Press 'F1' to show help";

char g_strokeTexFileName[256];

//Switches for effects
bool g_showHelp = true; // could hide the help text
bool g_renderNPR = true; // need Non-Photographic Rendering
bool g_renderColor = false;
bool g_renderWireFrame = false;
bool g_randomWiggling = false;
bool g_alphaTransition = true;
bool g_widthTransition = true;

//total number of objs, read from config.ini
int  g_ObjNum;

//Rect for text
const RECT screenRect={0, 0, WIDTH,HEIGHT};

//Initial stroke width
float g_strokeWidth = 0.1f;

//Device
IDirect3DDevice9* Device = 0;

// Containers for mesh info
ID3DXMesh**		g_meshes = NULL;
ID3DXBuffer**	g_adjBuffer = NULL;
D3DXMATRIX*		g_worldMatrices = NULL;
D3DXVECTOR4*	g_meshColors = NULL;
ID3DXFont*		g_font = NULL;

// variables for shaders
IDirect3DVertexShader9* ToonShader = 0;
ID3DXConstantTable* ToonConstTable = 0;

D3DXMATRIX ProjMatrix;

IDirect3DTexture9* ShadeTex  = 0;
IDirect3DTexture9* SilhouetteTex = 0;

D3DXHANDLE ToonWorldViewHandle     = 0;
D3DXHANDLE ToonWorldViewProjHandle = 0;
D3DXHANDLE ToonColorHandle    = 0;
D3DXHANDLE ToonLightDirHandle = 0;

IDirect3DVertexShader9* OutlineShader = 0;
ID3DXConstantTable* OutlineConstTable = 0;

D3DXHANDLE OutlineWorldViewHandle = 0;
D3DXHANDLE OutlineProjHandle = 0;
D3DXHANDLE OutlineStrokeWidth = 0;

// My shading effect handler
CelShadingHandler*	celShadingHandler;

// Silhouettes info container
CelSilhouette**		celSilhouettes;

// Global functions
void LoadConfigFile();
bool SetupFont();
void RenderFont(const char* str, RECT rect);
bool Setup();
void Cleanup();
bool Display(float timeDelta);

// WinMain
int WINAPI WinMain(HINSTANCE hinstance,
				   HINSTANCE prevInstance, 
				   PSTR cmdLine,
				   int showCmd)
{
	if(!d3d::InitD3D(hinstance,
		WIDTH, HEIGHT, true, D3DDEVTYPE_HAL, &Device))
	{
		::MessageBox(0, "InitD3D() - FAILED", 0, 0);
		return 0;
	}

	if(!Setup())
	{
		::MessageBox(0, "Setup() - FAILED", 0, 0);
		return 0;
	}

	d3d::EnterMsgLoop( Display );

	Cleanup();

	Device->Release();

	return 0;
}

// WndProc
LRESULT CALLBACK d3d::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch( msg )
	{
	case WM_DESTROY:
		::PostQuitMessage(0);
		break;

	case WM_KEYDOWN:
		if( wParam == VK_ESCAPE )
			::DestroyWindow(hwnd);
		else if( wParam == VK_F1)
			g_showHelp = !g_showHelp;
		else if( wParam == VK_F2 )
			g_renderNPR = !g_renderNPR;
		else if( wParam == VK_F3 )
			g_renderColor = !g_renderColor;
		else if( wParam == VK_F4 )
			g_renderWireFrame = !g_renderWireFrame;
		else if( wParam == VK_F5 )
			g_randomWiggling = !g_randomWiggling;
		else if( wParam == VK_F6 )
			g_alphaTransition = !g_alphaTransition;
		else if( wParam == VK_F7 )
			g_widthTransition = !g_widthTransition;
		else if( wParam == 65 )
		{
			g_strokeWidth += 0.05f;
		}
		else if( wParam == 90 )
		{
			g_strokeWidth -= 0.05f;
			g_strokeWidth = max(0.0f, g_strokeWidth);
		}

		break;
	}
	return ::DefWindowProc(hwnd, msg, wParam, lParam);
}

bool SetupFont()
{
	D3DXCreateFont(	Device,     //D3D Device
					22,               //Font height
					0,                //Font width
					FW_NORMAL,        //Font Weight
					1,                //MipLevels
					false,            //Italic
					DEFAULT_CHARSET,  //CharSet
					OUT_DEFAULT_PRECIS, //OutputPrecision
					ANTIALIASED_QUALITY, //Quality
					DEFAULT_PITCH|FF_DONTCARE,//PitchAndFamily
					"Arial",          //pFacename,
					&g_font);         //ppFont

	return true;
}

void RenderFont(const char* str, RECT rect)
{	
	g_font->DrawText(	NULL,        //pSprite
						str,  //pString
						-1,          //Count
						&rect,  //pRect
						DT_LEFT|DT_NOCLIP,//Format,
						0xFF000000); //Color
}

//
// Framework functions
//
bool Setup()
{
	HRESULT hr = 0;

	LoadConfigFile();

	SetupFont();

	celSilhouettes		= new CelSilhouette*[g_ObjNum];
	celShadingHandler	= new CelShadingHandler(Device);
	
	for(int i=0; i<g_ObjNum; ++i)
		celSilhouettes[i] = new CelSilhouette(Device, g_meshes[i], g_adjBuffer[i]);

	// toon shader
	ID3DXBuffer* toonCompiledCode = 0;
	ID3DXBuffer* toonErrorBuffer  = 0;

	hr = D3DXCompileShaderFromFile(
		"toon.txt",
		0,
		0,
		"Main",
		"vs_1_1",
		D3DXSHADER_DEBUG, 
		&toonCompiledCode,
		&toonErrorBuffer,
		&ToonConstTable);

	if( toonErrorBuffer )
	{
		::MessageBox(0, (char*)toonErrorBuffer->GetBufferPointer(), 0, 0);
		d3d::Release<ID3DXBuffer*>(toonErrorBuffer);
	}

	if(FAILED(hr))
	{
		::MessageBox(0, "D3DXCompileShaderFromFile() - FAILED", 0, 0);
		return false;
	}

	hr = Device->CreateVertexShader(
		(DWORD*)toonCompiledCode->GetBufferPointer(),
		&ToonShader);

	if(FAILED(hr))
	{
		::MessageBox(0, "CreateVertexShader - FAILED", 0, 0);
		return false;
	}

	d3d::Release<ID3DXBuffer*>(toonCompiledCode);

	//Outline shader
	ID3DXBuffer* outlineCompiledCode = 0;
	ID3DXBuffer* outlineErrorBuffer  = 0;

	hr = D3DXCompileShaderFromFile(
		"myOutline.txt",
		0,
		0,
		"Main",
		"vs_1_1",
		D3DXSHADER_DEBUG, 
		&outlineCompiledCode,
		&outlineErrorBuffer,
		&OutlineConstTable);

	if( outlineErrorBuffer )
	{
		::MessageBox(0, (char*)outlineErrorBuffer->GetBufferPointer(), 0, 0);
		d3d::Release<ID3DXBuffer*>(outlineErrorBuffer);
	}

	if(FAILED(hr))
	{
		::MessageBox(0, "D3DXCompileShaderFromFile() - FAILED", 0, 0);
		return false;
	}

	hr = Device->CreateVertexShader(
		(DWORD*)outlineCompiledCode->GetBufferPointer(),
		&OutlineShader);

	if(FAILED(hr))
	{
		::MessageBox(0, "CreateVertexShader - FAILED", 0, 0);
		return false;
	}

	d3d::Release<ID3DXBuffer*>(outlineCompiledCode);


	D3DXCreateTextureFromFile(Device, "toonshade.bmp", &ShadeTex);
	D3DXCreateTextureFromFile(Device, g_strokeTexFileName, &SilhouetteTex);

	D3DXCreateTextureFromFileEx(Device, g_strokeTexFileName, 
								D3DX_DEFAULT, D3DX_DEFAULT, 1, 0, 
								D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, 
								D3DX_FILTER_POINT, D3DX_FILTER_POINT, 
								0xFFFFFFFF, NULL, NULL, &SilhouetteTex);

	Device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
	Device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
	Device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);

	ToonWorldViewHandle     = ToonConstTable->GetConstantByName(0, "WorldViewMatrix");
	ToonWorldViewProjHandle = ToonConstTable->GetConstantByName(0, "WorldViewProjMatrix");
	ToonColorHandle         = ToonConstTable->GetConstantByName(0, "Color");
	ToonLightDirHandle      = ToonConstTable->GetConstantByName(0, "LightDirection");

	OutlineWorldViewHandle = OutlineConstTable->GetConstantByName(0, "WorldViewMatrix");
	OutlineProjHandle      = OutlineConstTable->GetConstantByName(0, "ProjMatrix");
	OutlineStrokeWidth	   = OutlineConstTable->GetConstantByName(0, "StrokeWidth");

	//
	// Set shader constants:
	//
	
	// Light direction:
	D3DXVECTOR4 directionToLight(-0.57f, 0.57f, -0.57f, 0.0f);

	ToonConstTable->SetVector(Device, ToonLightDirHandle, &directionToLight);
	ToonConstTable->SetDefaults(Device);
	OutlineConstTable->SetDefaults(Device);

	D3DXMatrixPerspectiveFovLH(&ProjMatrix, D3DX_PI * 0.25f, (float)WIDTH / (float)HEIGHT, 1.0f, 1000.0f);

	return true;
}

void Cleanup()
{
	for(int i=0; i<g_ObjNum; ++i)
	{
		d3d::Release(g_meshes[i]);
		d3d::Release(g_adjBuffer[i]);

		delete [] g_worldMatrices;
		delete [] g_meshColors;
	}

	delete [] g_meshes;
	delete [] g_adjBuffer;

	d3d::Release<IDirect3DTexture9*>(ShadeTex);
	d3d::Release<IDirect3DVertexShader9*>(ToonShader);
	d3d::Release<ID3DXConstantTable*>(ToonConstTable);
	d3d::Release<IDirect3DVertexShader9*>(OutlineShader);
	d3d::Release<ID3DXConstantTable*>(OutlineConstTable);

	for(int i=0; i<g_ObjNum; ++i)
	{
		if(celSilhouettes[i])
			delete celSilhouettes[i];
	}

	if(celSilhouettes)
		delete [] celSilhouettes;

	if(celShadingHandler)
		delete celShadingHandler;

	if(g_font)
	{
		g_font->Release();
		g_font=NULL;
	}
}

bool Display(float timeDelta)
{
	if( Device )
	{
		static float angle  = (3.0f * D3DX_PI) / 2.0f;
		static float height = 5.0f;
	
		if( ::GetAsyncKeyState(VK_LEFT) & 0x8000f )
			angle -= 0.5f * timeDelta;

		if( ::GetAsyncKeyState(VK_RIGHT) & 0x8000f )
			angle += 0.5f * timeDelta;

		if( ::GetAsyncKeyState(VK_UP) & 0x8000f )
			height += 5.0f * timeDelta;

		if( ::GetAsyncKeyState(VK_DOWN) & 0x8000f )
			height -= 5.0f * timeDelta;

		D3DXVECTOR3 position( cosf(angle) * 7.0f, height, sinf(angle) * 7.0f );
		
		D3DXVECTOR3 target(0.0f, 0.0f, 0.0f);
		D3DXVECTOR3 up(0.0f, 1.0f, 0.0f);
		D3DXMATRIX view;

		D3DXMatrixLookAtLH(&view, &position, &target, &up);

		//
		// Render
		//

		Device->Clear(0, 0, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xffffffff, 1.0f, 0);
		Device->BeginScene();

		if(g_showHelp)
			RenderFont(HELP_STRING, screenRect);
		else
			RenderFont(HIDE_STRING, screenRect);

		// Draw Cartoon
		if(g_renderWireFrame)
			Device->SetRenderState(D3DRS_FILLMODE , D3DFILL_WIREFRAME );
		else
			Device->SetRenderState(D3DRS_FILLMODE , D3DFILL_SOLID );

		Device->SetVertexShader(ToonShader);
		Device->SetTexture(0, ShadeTex);

		D3DXMATRIX worldView;
		D3DXMATRIX worldViewProj;

		for(int i = 0; i < g_ObjNum; i++)
		{
			worldView = g_worldMatrices[i] * view;
			worldViewProj = g_worldMatrices[i] * view * ProjMatrix;
 
			ToonConstTable->SetMatrix(
				Device, 
				ToonWorldViewHandle,
				&worldView);

			ToonConstTable->SetMatrix(
				Device, 
				ToonWorldViewProjHandle,
				&worldViewProj);

			ToonConstTable->SetVector(
				Device,
				ToonColorHandle,
				&g_meshColors[i]);
			
			if(g_renderColor)
				g_meshes[i]->DrawSubset(0);
		}

		// Draw Outlines.

		Device->SetRenderState(D3DRS_ALPHABLENDENABLE,TRUE);
		Device->SetRenderState(D3DRS_SRCBLEND,D3DBLEND_SRCALPHA);
		Device->SetRenderState(D3DRS_DESTBLEND,D3DBLEND_INVSRCALPHA);

		Device->SetTextureStageState( 0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE ); 
		Device->SetTextureStageState( 0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE );
		Device->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_MODULATE );
		
		Device->SetRenderState(D3DRS_FILLMODE , D3DFILL_SOLID );

		if(g_renderNPR)
		{
			Device->SetVertexShader(OutlineShader);
			Device->SetTexture(0, SilhouetteTex);
			Device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
			
			//disable Z BUFFER, render silhouettes
			Device->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
		}

		for(int i = 0; i < g_ObjNum; i++)
		{
			worldView = g_worldMatrices[i] * view;

			if(g_renderNPR)
				celShadingHandler->process(celSilhouettes[i], &worldView, &ProjMatrix);

			OutlineConstTable->SetMatrix(
				Device, 
				OutlineWorldViewHandle,
				&worldView);

			OutlineConstTable->SetMatrix(
				Device, 
				OutlineProjHandle,
				&ProjMatrix);

			OutlineConstTable->SetFloat(
				Device, 
				OutlineStrokeWidth, 
				g_strokeWidth);
			
			if(g_renderNPR)
				celSilhouettes[i]->render();
		}

		if(g_renderNPR)
		{
			Device->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE);
			Device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
		}

		Device->SetRenderState(D3DRS_ALPHABLENDENABLE,FALSE);   
		Device->SetRenderState(D3DRS_SRCBLEND,D3DBLEND_ONE);   
		Device->SetRenderState(D3DRS_DESTBLEND,D3DBLEND_ZERO);

		Device->EndScene();

		Device->Present(0, 0, 0, 0);
	}
	return true;
}

void LoadConfigFile()
{
	::GetPrivateProfileString("Config", "StrokeTexture", "", g_strokeTexFileName, 256, CONFIG_FILE_NAME);

	g_ObjNum = ::GetPrivateProfileInt("Config", "ObjNum", 0, CONFIG_FILE_NAME);

	// Create geometry and compute corresponding world matrix and color
	// for each mesh.
	g_meshes		= new ID3DXMesh*[g_ObjNum];
	g_adjBuffer		= new ID3DXBuffer*[g_ObjNum];
	g_worldMatrices	= new D3DXMATRIX[g_ObjNum];
	g_meshColors	= new D3DXVECTOR4[g_ObjNum];

	for(int i=0; i<g_ObjNum; ++i)
		g_meshColors[i] = D3DXVECTOR4(1.0, 1.0, 0, 1.0);// default Color for mesh

	for(int i=0; i<g_ObjNum; ++i)
	{
		float offsetX,  offsetY, offsetZ;

		char idx[32];
		itoa(i, idx, 10);

		char objIdx[64] = "Obj";
		strcat(objIdx, idx);

		char objType[64];

		::GetPrivateProfileString(objIdx, "Geometry", "", objType, 256, CONFIG_FILE_NAME);

		char tmp[32];

		if(strcmp(objType, "Cylinder") == 0)
		{
			float radius1, radius2;
			int length, slice, stack;

			::GetPrivateProfileString(objIdx, "Radius1", "", tmp, 32, CONFIG_FILE_NAME);
			radius1 = atof(tmp);

			::GetPrivateProfileString(objIdx, "Radius2", "", tmp, 32, CONFIG_FILE_NAME);
			radius2 = atof(tmp);

			length = ::GetPrivateProfileInt(objIdx, "Length", 0, CONFIG_FILE_NAME);
			slice = ::GetPrivateProfileInt(objIdx, "Slice", 0, CONFIG_FILE_NAME);
			stack = ::GetPrivateProfileInt(objIdx, "Stack", 0, CONFIG_FILE_NAME);

			D3DXCreateCylinder(Device, radius1, radius2, length, slice, stack, &g_meshes[i], &g_adjBuffer[i]);
		}
		else if(strcmp(objType, "Box") == 0)
		{
			float width, height, depth;

			::GetPrivateProfileString(objIdx, "Width", "", tmp, 32, CONFIG_FILE_NAME);
			width = atof(tmp);

			::GetPrivateProfileString(objIdx, "Height", "", tmp, 32, CONFIG_FILE_NAME);
			height = atof(tmp);

			::GetPrivateProfileString(objIdx, "Depth", "", tmp, 32, CONFIG_FILE_NAME);
			depth = atof(tmp);

			D3DXCreateBox(Device, width, height, depth, &g_meshes[i], &g_adjBuffer[i]);
		}
		else if(strcmp(objType, "Sphere") == 0)
		{
			float radius;
			int slice, stack;

			::GetPrivateProfileString(objIdx, "Radius", "", tmp, 32, CONFIG_FILE_NAME);
			radius = atof(tmp);

			slice = ::GetPrivateProfileInt(objIdx, "Slice", 0, CONFIG_FILE_NAME);
			stack = ::GetPrivateProfileInt(objIdx, "Stack", 0, CONFIG_FILE_NAME);

			D3DXCreateSphere(Device, radius, slice, stack, &g_meshes[i], &g_adjBuffer[i]);

		}
		else if(strcmp(objType, "Torus") == 0)
		{
			float innerR, outerR;
			int sides, rings;

			::GetPrivateProfileString(objIdx, "InnerR", "", tmp, 32, CONFIG_FILE_NAME);
			innerR = atof(tmp);

			::GetPrivateProfileString(objIdx, "OutterR", "", tmp, 32, CONFIG_FILE_NAME);
			outerR = atof(tmp);

			sides = ::GetPrivateProfileInt(objIdx, "Sides", 0, CONFIG_FILE_NAME);
			rings = ::GetPrivateProfileInt(objIdx, "Rings", 0, CONFIG_FILE_NAME);

			D3DXCreateTorus(Device, innerR, outerR, sides, rings, &g_meshes[i], &g_adjBuffer[i]);
		}
		else if(strcmp(objType, "TeaPot") == 0)
		{
			D3DXCreateTeapot(Device, &g_meshes[i], &g_adjBuffer[i]);
		}

		//Translations
		::GetPrivateProfileString(objIdx, "PosX", "", tmp, 32, CONFIG_FILE_NAME);
		offsetX = atof(tmp);

		::GetPrivateProfileString(objIdx, "PosY", "", tmp, 32, CONFIG_FILE_NAME);
		offsetY = atof(tmp);

		::GetPrivateProfileString(objIdx, "PosZ", "", tmp, 32, CONFIG_FILE_NAME);
		offsetZ = atof(tmp);

		D3DXMatrixTranslation(&(g_worldMatrices[i]), offsetX,  offsetY, offsetZ);
	}
}
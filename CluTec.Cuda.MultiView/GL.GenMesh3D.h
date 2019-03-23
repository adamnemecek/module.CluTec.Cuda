////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      GL.GenMesh3D.h
//
// summary:   Declares the gl. generate mesh 3D class
//
//            Copyright (c) 2019 by Christian Perwass.
//
//            This file is part of the CluTecLib library.
//
//            The CluTecLib library is free software: you can redistribute it and / or modify
//            it under the terms of the GNU Lesser General Public License as published by
//            the Free Software Foundation, either version 3 of the License, or
//            (at your option) any later version.
//
//            The CluTecLib library is distributed in the hope that it will be useful,
//            but WITHOUT ANY WARRANTY; without even the implied warranty of
//            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//            GNU Lesser General Public License for more details.
//
//            You should have received a copy of the GNU Lesser General Public License
//            along with the CluTecLib library.
//            If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_runtime.h"
#include "CluTec.Cuda.Base/Conversion.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.Cuda.Base/Gl.BufferMap.h"
#include "CluTec.ImgProc/Camera.StereoPinhole.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace GL
		{
			namespace GenMesh3D
			{
				using TValue = float;
				using TPixelMap3D = Clu::TPixel_RGBA_Single;

				struct _SParameter
				{
					int iVexCountX;
					int iVexCountY;

					Clu::_SVector<TValue, 3> _vCenter;
					TValue _fScale;
				};

				struct SParameter : public _SParameter
				{
					TPixelMap3D pixMin, pixMax;

					SParameter()
					{
						iVexCountX = 0;
						iVexCountY = 0;
					}

					void Set(int iCntX, int iCntY, const TPixelMap3D& _pixMin, const TPixelMap3D& _pixMax)
					{
						iVexCountX = iCntX;
						iVexCountY = iCntY;

						pixMin = _pixMin;
						pixMax = _pixMax;
					}
				};



				class CDriver : public Clu::Cuda::CKernelDriverBase
				{
				public:
#ifdef _DEBUG
					static const size_t NumberOfRegisters = 17;
#else
					static const size_t NumberOfRegisters = 17;
#endif
				private:
					SParameter m_xPars;

				public:
					CDriver()
						: CKernelDriverBase("Generate OpenGL 3D Mesh")
					{}

					~CDriver()
					{}

					void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat, const SParameter& xPars);

					template<typename TVertex>
					void Process(Clu::Cuda::GL::CBufferMap<TVertex>& xGlBuffer, const Clu::Cuda::_CDeviceSurface& surfMap3D);
				};
			} // GenMesh3D
		} // GL
	} // Cuda
} // Clu


////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Filter.Sobel.h
//
// summary:   Declares the filter. sobel class
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
#include "CluTec.Math/Static.Matrix.h"
#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.ImgProc/Camera.Pinhole.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Filter
		{
			namespace Sobel
			{
				enum class EConfig
				{
					Patch_3x3 = 0,
					Patch_5x5,
					Patch_7x7,
					Patch_9x9,
				};


				template<EConfig t_eConfig>
				struct SConfig
				{};

				template<> struct SConfig<EConfig::Patch_3x3>
				{
					static const int Radius = 1;
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
					static const int NumberOfRegisters = 12;
				};

				template<> struct SConfig<EConfig::Patch_5x5>
				{
					static const int Radius = 2;
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
					static const int NumberOfRegisters = 12;
				};

				template<> struct SConfig<EConfig::Patch_7x7>
				{
					static const int Radius = 3;
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
					static const int NumberOfRegisters = 12;
				};

				template<> struct SConfig<EConfig::Patch_9x9>
				{
					static const int Radius = 4;
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
					static const int NumberOfRegisters = 12;
				};



				struct _SParameter
				{
					float fGamma;
					float fScale;
				};


				struct SParameter : public _SParameter
				{
					EConfig eConfig;

					SParameter()
					{
						eConfig = EConfig::Patch_3x3;
					}

					void Set(float _fGamma, float _fScale, EConfig _eConfig)
					{
						fGamma = _fGamma;
						fScale = _fScale;
						eConfig = _eConfig;
					}
				};

				class CDriver : public CKernelDriverBase
				{
				public:
					static const int KID_ConvolveH = 0;
					static const int KID_ConvolveV = 1;
					static const int KID_AbsGrad = 2;

				private:
					SParameter m_xPars;

				private:
					template<EConfig t_eConfig>
					void _DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

					template<typename TPixelSrc, typename TPixelSum>
					void _SelectConfig(Clu::Cuda::_CDeviceSurface& xImageTrg
						, const Clu::Cuda::_CDeviceSurface& xImageSrc
						, Clu::Cuda::CDeviceSurface& xImageTempH
						, Clu::Cuda::CDeviceSurface& xImageTempV);

					template<typename TPixelSrc, typename TPixelSum, typename TConfig>
					void _DoProcess(Clu::Cuda::_CDeviceSurface& xImageTrg
						, const Clu::Cuda::_CDeviceSurface& xImageSrc
						, Clu::Cuda::CDeviceSurface& xImageTempH
						, Clu::Cuda::CDeviceSurface& xImageTempV);

				public:
					CDriver()
						: CKernelDriverBase("Sobel Filter")
					{}

					~CDriver()
					{}

					void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
						, const SParameter& xPars);

					void Process(Clu::Cuda::_CDeviceSurface& xImageTrg
						, const Clu::Cuda::_CDeviceSurface& xImageSrc
						, Clu::Cuda::CDeviceSurface& xImageTempH
						, Clu::Cuda::CDeviceSurface& xImageTempV);



				};
			} // Sobel
		} // Filter
	} // Cuda
} // Clu


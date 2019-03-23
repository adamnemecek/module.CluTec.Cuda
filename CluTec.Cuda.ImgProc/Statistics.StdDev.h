////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Statistics.StdDev.h
//
// summary:   Declares the statistics. standard development class
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
		namespace Statistics
		{
			namespace StdDev
			{
				enum class EConfig
				{
					Patch_16x16 = 0,
					Patch_11x11,
					Patch_9x9,
					Patch_7x7,
					Patch_5x5,
					Patch_3x3,
				};


				template<EConfig t_eConfig>
				struct SConfig
				{};

				template<> struct SConfig<EConfig::Patch_16x16>
				{
					static const int PatchSizeX = 16;
					static const int PatchSizeY = 16;
					static const int PatchCountY_Pow2 = 3;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};

				template<> struct SConfig<EConfig::Patch_11x11>
				{
					static const int PatchSizeX = 11;
					static const int PatchSizeY = 11;
					static const int PatchCountY_Pow2 = 3;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};

				template<> struct SConfig<EConfig::Patch_9x9>
				{
					static const int PatchSizeX = 9;
					static const int PatchSizeY = 9;
					static const int PatchCountY_Pow2 = 2;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};

				template<> struct SConfig<EConfig::Patch_7x7>
				{
					static const int PatchSizeX = 7;
					static const int PatchSizeY = 7;
					static const int PatchCountY_Pow2 = 2;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};

				template<> struct SConfig<EConfig::Patch_5x5>
				{
					static const int PatchSizeX = 5;
					static const int PatchSizeY = 5;
					static const int PatchCountY_Pow2 = 2;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};

				template<> struct SConfig<EConfig::Patch_3x3>
				{
					static const int PatchSizeX = 3;
					static const int PatchSizeY = 3;
					static const int PatchCountY_Pow2 = 2;
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int NumberOfRegisters = 53;
				};


				struct _SParameter
				{
					float fScale;
				};


				struct SParameter : public _SParameter
				{
					EConfig eConfig;

					SParameter()
					{
						fScale = 1.0f;
						eConfig = EConfig::Patch_5x5;
					}

					void Set(float _fScale, EConfig _eConfig)
					{
						fScale = _fScale;
						eConfig = _eConfig;
					}
				};

				class CDriver : public CKernelDriverBase
				{
				private:
					SParameter m_xPars;

				private:
					template<EConfig t_eConfig>
					void _DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

					template<typename TPixel>
					void _DoProcess(Clu::Cuda::_CDeviceSurface& xImageTrg
						, const Clu::Cuda::_CDeviceSurface& xImageSrc);

				public:
					CDriver()
						: CKernelDriverBase("Statistics Mean Filter")
					{}

					~CDriver()
					{}

					void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
						, const SParameter& xPars);

					void Process(Clu::Cuda::_CDeviceSurface& xImageTrg
						, const Clu::Cuda::_CDeviceSurface& xImageSrc);



				};

			}

		} // ImgProc
	} // Cuda
} // Clu


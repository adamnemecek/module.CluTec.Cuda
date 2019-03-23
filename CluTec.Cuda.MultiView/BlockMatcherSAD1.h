////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherSAD1.h
//
// summary:   Declares the block matcher sad 1 class
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

#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"

namespace Clu
{
	namespace Cuda
	{
		namespace BlockMatcherSAD1
		{
			using TDisp = _SDisparityConfig::TDisp;

			enum class EConfig
			{
				Patch_16x16 = 0,
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


			struct _SParameter
			{
				int iOffset;
				int iDispRange;
				float fSadThresh;
				int iMinDeltaThresh;
				float fGradThresh;

				// Calculated values
				int _iBlockOffsetX;
				int _iBlockOffsetY;
				int _iRightToLeftOffsetX;
			};


			struct SParameter : public _SParameter
			{
				EConfig eConfig;

				void Set(int _iOffset, int _iDispRange, int _iMinDeltaThresh, float _fSadThresh, float _fGradThresh, EConfig _eConfig)
				{
					iOffset = _iOffset;
					iDispRange = _iDispRange;
					iMinDeltaThresh = _iMinDeltaThresh;
					fSadThresh = _fSadThresh;
					fGradThresh = _fGradThresh;
					eConfig = _eConfig;
				}
			};

			class CDriver : public Clu::Cuda::CKernelDriverBase
			{
			public:
				using TPixelDisp = _SDisparityConfig::TPixelEx;

			private:
				SParameter m_xPars;

			private:
				template<EConfig t_eConfig>
				void _DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

				template<typename TPixelDisp, typename TPixelSrc>
				void _DoProcess();

			public:
				CDriver()
					: CKernelDriverBase("Block Matcher SAD 1")
				{}

				~CDriver()
				{}


				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,	const SParameter& xPars);

				void Process(Clu::Cuda::CDeviceSurface& xImageDisp
					, const Clu::Cuda::CDeviceSurface& xImageL
					, const Clu::Cuda::CDeviceSurface& xImageR);
			};


		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherAW2.h
//
// summary:   Declares the block matcher a w 2 class
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
		namespace BlockMatcherAW2
		{
			using TPixelDisp = _SDisparityConfig::TPixelEx;
			using TDisp = _SDisparityConfig::TDisp;

			enum class EConfig
			{
				Patch_15x15 = 0,
				Patch_9x9,
			};

			template<EConfig t_eConfig>
			struct SConfig
			{};

			template<> struct SConfig<EConfig::Patch_15x15>
			{
				static const int SubPatchSizeX = 5;
				static const int SubPatchSizeY = 5;
				static const int DispPerThread = 2;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 56;
			};

			template<> struct SConfig<EConfig::Patch_9x9>
			{
				static const int SubPatchSizeX = 3;
				static const int SubPatchSizeY = 3;
				static const int DispPerThread = 2;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 56;
			};

			struct _SParameter
			{
				int iOffset;
				int iDispRange;
				int iMinDeltaThresh;
				float fSadThresh;
				float fGradThresh;
				int iUseDispInput;

				// Calculated values
				int _iBlockOffsetX;
				int _iBlockOffsetY;
				int _iRightToLeftOffsetX;
			};

			struct SParameter : public _SParameter
			{
				EConfig eConfig;

				void Set(int _iOffset, int _iDispRange, int _iMinDeltaThresh, float _fSadThresh, float _fGradThresh, int _iUseDispInput, EConfig _eConfig)
				{
					iOffset = _iOffset;
					iDispRange = _iDispRange;
					iMinDeltaThresh = _iMinDeltaThresh;
					fSadThresh = _fSadThresh;
					fGradThresh = _fGradThresh;
					iUseDispInput = _iUseDispInput;
					eConfig = _eConfig;
				}
			};

			class CDriver : public Clu::Cuda::CKernelDriverBase
			{
			public:

			private:
				SParameter m_xPars;

			private:
				template<EConfig t_eConfig>
				void _DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

				template<typename TPixelDisp, typename TPixelSrc>
				void _DoProcess();

			public:
				CDriver() 
					: Clu::Cuda::CKernelDriverBase("Block Matcher Adaptive Window 2")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,	const SParameter& xPars);

				void Process(Clu::Cuda::_CDeviceSurface& xImageDisp
					, const Clu::Cuda::_CDeviceSurface& xImageL
					, const Clu::Cuda::_CDeviceSurface& xImageR
					, const Clu::Cuda::_CDeviceSurface& xImageDispInit);
			};


		}
	}
}
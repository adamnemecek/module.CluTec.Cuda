////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherNCC.h
//
// summary:   Declares the block matcher ncc class
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
		namespace BlockMatcherNCC
		{
			enum class EConfig
			{
				Patch_11x11 = 0,
				Patch_9x9,
				Patch_5x5,
			};

			template<EConfig t_eConfig>
			struct SConfig
			{};

			template<> struct SConfig<EConfig::Patch_11x11>
			{
				static const int PatchSizeX = 11;
				static const int PatchSizeY = 11;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 63;
			};

			template<> struct SConfig<EConfig::Patch_9x9>
			{
				static const int PatchSizeX = 9;
				static const int PatchSizeY = 9;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 63;
			};

			template<> struct SConfig<EConfig::Patch_5x5>
			{
				static const int PatchSizeX = 5;
				static const int PatchSizeY = 5;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 63;
			};

			struct _SParameter
			{
				_SDisparityConfig xDispConfig;

				int iMinDeltaThresh;
				float fNccThresh;
				float fGradThresh;
				int iUseDispInput;

				// Calculated values
				int _iBlockOffsetX;
				int _iBlockOffsetY;
				int _iIsLeftToRight;
			};

			struct SParameter : public _SParameter
			{
				EConfig eConfig;
				bool bLeftToRight;

				void Set(const _SDisparityConfig& _xDispConfig, int _iMinDeltaThresh, float _fSadThresh, float _fGradThresh, int _iUseDispInput, bool _bLeftToRight, EConfig _eConfig)
				{
					xDispConfig = _xDispConfig;

					iMinDeltaThresh = _iMinDeltaThresh;
					fNccThresh = _fSadThresh;
					fGradThresh = _fGradThresh;
					iUseDispInput = _iUseDispInput;
					bLeftToRight = _bLeftToRight;
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
					: Clu::Cuda::CKernelDriverBase("Block Matcher Adaptive Window 1")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,	const SParameter& xPars);

				void Process(Clu::Cuda::_CDeviceSurface& xImageDisp
					, const Clu::Cuda::_CDeviceSurface& xImageL
					, const Clu::Cuda::_CDeviceSurface& xImageR
					, const Clu::Cuda::_CDeviceSurface& xImageDispInit
					, const Clu::Cuda::_CDeviceSurface& xImageDebug);
			};


		}
	}
}
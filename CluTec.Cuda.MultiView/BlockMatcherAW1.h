////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherAW1.h
//
// summary:   Declares the block matcher a w 1 class
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
		namespace BlockMatcherAW1
		{
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
				static const int SubPatchCountY_Pow2 = 4;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 56;
			};

			template<> struct SConfig<EConfig::Patch_9x9>
			{
				static const int SubPatchSizeX = 3;
				static const int SubPatchSizeY = 3;
				static const int SubPatchCountY_Pow2 = 4;
				static const int WarpsPerBlockX = 1;
				static const int WarpsPerBlockY = 1;
				static const int NumberOfRegisters = 56;
			};

			enum class ESimType
			{
				AbsoluteDifference = 0,
				SignedProduct,
			};


			struct SSimType_AbsoluteDifference {};
			struct SSimType_SignedProduct {};


			struct _SParameter
			{
				_SDisparityConfig xDispConfig;

				float fMinDeltaThresh;
				float fSimThresh;
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
				ESimType eSimType;

				bool bLeftToRight;

				void Set(const _SDisparityConfig& _xDispConfig, float _fMinDeltaThresh, float _fSimThresh
					, float _fGradThresh, int _iUseDispInput, bool _bLeftToRight
					, EConfig _eConfig, ESimType _eSimType)
				{
					xDispConfig = _xDispConfig;

					fMinDeltaThresh = _fMinDeltaThresh;
					fSimThresh = _fSimThresh;
					fGradThresh = _fGradThresh;
					iUseDispInput = _iUseDispInput;
					bLeftToRight = _bLeftToRight;
					eConfig = _eConfig;
					eSimType = _eSimType;
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
				void _SelectPatchSize();

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_ePatchSize>
				void _SelectSimType();

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_ePatchSize, typename TSimType>
				void _SelectUseDispInput();

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
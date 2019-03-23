////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      CrossCheckDisparityMaps.h
//
// summary:   Declares the cross check disparity maps class
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
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"

namespace Clu
{
	namespace Cuda
	{
		namespace CrossCheckDisparityMaps
		{
			using TPixelDisp = _SDisparityConfig::TPixelEx;
			using TDisp = _SDisparityConfig::TDisp;
			using TPixelRange = Clu::CastToPixel<Clu::EPixelType::Lum, Clu::EDataType::UInt32>::Type;
			using TRange = TPixelRange::TData;

			struct _SParameter
			{
				_SDisparityConfig xDispConfig;
				int iDispDeltaThresh;
				int iIsLeftToRight;
			};


			struct SParameter : public _SParameter
			{
				TDisp uDispMin, uDispMax;

				void Set(const _SDisparityConfig& _xDispConfig, int _iDispDeltaThresh, int _iIsLeftToRight)
				{
					xDispConfig = _xDispConfig;
					iDispDeltaThresh = _iDispDeltaThresh;
					iIsLeftToRight = _iIsLeftToRight;
				}

				void ResetRange()
				{
					uDispMin = TDisp(EDisparityId::Last);
					uDispMax = TDisp(EDisparityId::First);
				}
			};

			class CDriver : public Clu::Cuda::CKernelDriverBase
			{
			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 30;
#else
				static const size_t NumberOfRegisters = 30;
#endif

			private:
				Clu::CIImage m_imgRange;
				Clu::Cuda::CDeviceImage m_deviRange;

			protected:
				SParameter m_xPars;

			public:
				CDriver()
					: CKernelDriverBase("Cross Check Disparity Maps")
				{}

				~CDriver()
				{}

				const TDisp& DispMin()
				{
					return m_xPars.uDispMin;
				}

				const TDisp& DispMax()
				{
					return m_xPars.uDispMax;
				}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
					, const SParameter& xPars);

				void Process(Clu::Cuda::_CDeviceSurface& xDispResult
					, const Clu::Cuda::_CDeviceSurface& xDispLR
					, const Clu::Cuda::_CDeviceSurface& xDispRL);



			};

		} // ImgProc
	} // Cuda
} // Clu


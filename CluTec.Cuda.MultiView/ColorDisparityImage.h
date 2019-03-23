////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      ColorDisparityImage.h
//
// summary:   Declares the color disparity image class
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
#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace ColorDisparityImage
		{
			using TPixelColor = Clu::TPixel_RGBA_UInt8;
			using TPixelGray = Clu::TPixel_Lum_UInt16;
			using TPixelDisp = _SDisparityConfig::TPixel;
			using TPixelDispEx = _SDisparityConfig::TPixelEx;
			using TPixelDispF = _SDisparityConfig::TPixelF;
			using TDisp = _SDisparityConfig::TDisp;

			enum class EStyle
			{
				Analysis = 0,
				Gray,
				Color,
				DispAbs,
				DispRel,
			};

			struct _SParameter
			{
				TDisp uDispMin, uDispMax;
				TDisp uDispInfOffset;

				TPixelColor pixUnknown;
				TPixelColor pixCannotEvaluate;
				TPixelColor pixSaturated;
				TPixelColor pixNotSpecific;
				TPixelColor pixNotFound;
				TPixelColor pixNotUnique;
				TPixelColor pixInconsistent;
			};

			struct SParameter : public _SParameter
			{
				SParameter()
				{
					uDispMin = TDisp(EDisparityId::First);
					uDispMax = TDisp(EDisparityId::Last);

					pixUnknown			= Clu::Cuda::Make<TPixelColor>( 10, 100, 180, 255);
					pixCannotEvaluate	= Clu::Cuda::Make<TPixelColor>(  0, 150, 50, 255);
					pixSaturated		= Clu::Cuda::Make<TPixelColor>(  0, 200, 200, 255);
					pixNotSpecific		= Clu::Cuda::Make<TPixelColor>(200,   0, 200, 255);
					pixNotFound			= Clu::Cuda::Make<TPixelColor>(200,  50,  50, 255);
					pixNotUnique = Clu::Cuda::Make<TPixelColor>(240, 50, 150, 255);
					pixInconsistent = Clu::Cuda::Make<TPixelColor>(100, 10, 10, 255);
				}

				void Set(TDisp _uDispMin, TDisp _uDispMax, TDisp _uDispInfOffset)
				{
					uDispMin = _uDispMin;
					uDispMax = _uDispMax;
					uDispInfOffset = _uDispInfOffset;
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
					: CKernelDriverBase("Color Disparity Image")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat, const SParameter& xPars);

				template<typename TPixelDisp>
				void SelectStyle(Clu::Cuda::_CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle);

				void Process(Clu::Cuda::CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle);
				void Process(Clu::Cuda::_CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle);
			};

		} // ImgProc
	} // Cuda
} // Clu


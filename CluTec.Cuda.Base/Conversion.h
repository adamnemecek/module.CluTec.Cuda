////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Conversion.h
//
// summary:   Declares the conversion class
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
#include "Api.h"
#include "PixelTypeInfo.h"

#include "CluTec.Math/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		template<typename TPixel>
		__CUDA_HDI__ typename SPixelTypeInfo<TPixel>::TElement NormToRawPix(const typename SPixelTypeInfo<TPixel>::TNormalized& xNormPix)
		{
			using TElement = typename SPixelTypeInfo<TPixel>::TElement;
			using TComponent = typename TPixel::TData;

			TElement xRawPix;
			TComponent *pRaw = &xRawPix.x;
			const decltype(xNormPix.x)* pNorm = &xNormPix.x;

			for (int i = 0; i < TPixel::ChannelCount; ++i, ++pRaw, ++pNorm)
			{
				*pRaw = Clu::NormFloatTo<TComponent>(*pNorm);
			}

			return xRawPix;
		}

		template<typename TPixel>
		__CUDA_HDI__ typename TPixel NormFloatToColor(float fValue)
		{
			using TComponent = typename TPixel::TData;

			TPixel pixVal;

			//float fVal = 1.0f - min(min(fValue / 0.33f, 1.0f), min((fValue - 1.0f) / 0.05f, 1.0f));
			float fB = 1.0f - min(fValue / 0.7f, 1.0f);
			float fG = 1.0f - min(abs(fValue - 0.5f) / 0.59f, 1.0f);
			float fR = 1.0f - min(abs(fValue - 0.8f) / 0.7f, 1.0f);
			
			float fMax = max(max(fR, fG), fB);

			pixVal.b() = Clu::NormFloatTo<TComponent>(fB / fMax);
			pixVal.g() = Clu::NormFloatTo<TComponent>(fG / fMax);
			pixVal.r() = Clu::NormFloatTo<TComponent>(fR / fMax);
			pixVal.a() = Clu::NumericLimits<TComponent>::Max();

			return pixVal;
		}


		template<typename TResult>
		__CUDA_HDI__ TResult ConvertTo(const uchar1& xValue);



		template<typename TElement, typename TComp>
		__CUDA_HDI__ TElement Make(TComp xVal1);

		template<typename TElement, typename TComp>
		__CUDA_HDI__ TElement Make(TComp xVal1, TComp xVal2, TComp xVal3, TComp xVal4);

		template<typename TElement>
		__CUDA_HDI__ TElement MakeZero();


		template<typename T>
		__CUDA_HDI__ void Assign(volatile T& xTrg, const T& xSrc);

		template<typename T>
		__CUDA_HDI__ void Assign(T& xTrg, const T& xSrc);

	}
}

#ifdef __CUDACC__
#include "Conversion_Impl.h"
#endif
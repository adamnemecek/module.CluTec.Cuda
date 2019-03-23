////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceImage.h
//
// summary:   Declares the device image class
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
#include <cstdint>

#include "CluTec.Types1/ImageFormat.h"
#include "CluTec.Types1/IImage.h"
#include "CluTec.Types1/ILayerImage.h"
#include "CluTec.Base/Exception.h"

#include "PixelTypeInfo.h"

#include "Api.h"
#include "DeviceArray1D.h"


namespace Clu
{
	namespace Cuda
	{
		class _CDeviceImage : protected _CDeviceArray1D<uint8_t>
		{
		public:
			using TDeviceObject = _CDeviceImage;

		protected:
			Clu::_SImageFormat m_xFormat;

		public:
			__CUDA_HDI__ const Clu::_SImageFormat& Format() const
			{
				return m_xFormat;
			}

			template<typename TPixel>
			__CUDA_HDI__ TPixel& At(int iX, int iY)
			{
				return ((TPixel *)m_pData)[GetPixelIndex(iX, iY)];
			}

			template<typename TPixel>
			__CUDA_HDI__ const TPixel& At(int iX, int iY) const
			{
				return ((TPixel *)m_pData)[GetPixelIndex(iX, iY)];
			}

			template<typename TPixel>
			__CUDA_HDI__ TPixel* PtrAt(int iX, int iY)
			{
				return &(((TPixel *)m_pData)[GetPixelIndex(iX, iY)]);
			}


			template<typename TPixel>
			__CUDA_HDI__ TPixel& At(int iIdx)
			{
				return ((TPixel *)m_pData)[iIdx];
			}

			__CUDA_HDI__ void* DataPointer()
			{
				return (void*) m_pData;
			}

			__CUDA_HDI__ const void* DataPointer() const
			{
				return (void*)m_pData;
			}

			__CUDA_HDI__ bool IsValid() const
			{
				return m_xFormat.IsValid() && _CDeviceArray1D::IsValid();
			}

			template<typename TPixel>
			__CUDA_HDI__ bool IsOfType() const
			{
				return m_xFormat.IsOfType<TPixel>();
			}

			template<typename TPixel>
			__CUDA_HDI__ bool IsOfType(const TPixel& xPix) const
			{
				return m_xFormat.IsOfType(xPix);
			}

			__CUDA_HDI__ bool IsEqualFormat(const Clu::_SImageFormat& xFormat) const
			{
				return m_xFormat == xFormat;
			}

			__CUDA_HDI__ bool IsEqualSize(const Clu::_SImageFormat& xFormat) const
			{
				return m_xFormat.IsEqualSize(xFormat);
			}

			__CUDA_HDI__ bool IsEqualType(const Clu::_SImageFormat& xFormat) const
			{
				return m_xFormat.IsEqualType(xFormat);
			}

			__CUDA_HDI__ bool IsEqualDataType(const Clu::_SImageFormat& xFormat) const
			{
				return m_xFormat.IsEqualDataType(xFormat);
			}

			__CUDA_HDI__ bool IsEqualPixelType(const Clu::_SImageFormat& xFormat) const
			{
				return m_xFormat.IsEqualPixelType(xFormat);
			}

			__CUDA_HDI__ size_t ByteCount() const
			{
				return m_xFormat.ByteCount();
			}

			__CUDA_HDI__ bool IsInside(int iX, int iY) const
			{
				return m_xFormat.IsInside(iX, iY);
			}

			__CUDA_HDI__ bool IsInside(int iX, int iY, int iBorderWidth, int iBorderHeight) const
			{
				return m_xFormat.IsInside(iX, iY, iBorderWidth, iBorderHeight);
			}

			__CUDA_HDI__ bool IsRectInside(int iLeftX, int iTopY, int iSizeX, int iSizeY) const
			{
				return m_xFormat.IsRectInside(iLeftX, iTopY, iSizeX, iSizeY);
			}


			__CUDA_HDI__ size_t GetPixelIndex(int iX, int iY) const
			{
				return m_xFormat.GetPixelIndex(iX, iY);
			}


			template<typename TElement>
			__CUDA_DI__ TElement Read2D(int iX, int iY) const
			{
				return At<TElement>(iX, iY);
			}

			template<typename TPixel>
			__CUDA_DI__ TPixel ReadPixel2D(int iX, int iY) const
			{
				return At<TPixel>(iX, iY);
			}

			template<typename TElement>
			__CUDA_DI__ void Read2D(TElement &xValue, int iX, int iY) const
			{
				xValue = At<TElement>(iX, iY);
			}

			template<typename TPixel>
			__CUDA_DI__ void ReadPixel2D(TPixel &xValue, int iX, int iY) const
			{
				xValue = At<TPixel>(iX, iY);
			}



			template<typename TElement>
			__CUDA_DI__ void Write2D(const TElement& xValue, int iX, int iY)
			{
				At<TElement>(iX, iY) = xValue;
			}

			template<typename TPixel>
			__CUDA_DI__ void WritePixel2D(const TPixel& xValue, int iX, int iY)
			{
				At<TPixel>(iX, iY) = xValue;
			}

			_CDeviceImage AsDeviceObject() const
			{
				return *this;
			}
		};


		class CDeviceImage : public _CDeviceImage
		{
		public:
			using TDeviceObject = _CDeviceImage;

		public:
			CDeviceImage();
			CDeviceImage(const Clu::_SImageFormat& xFormat);

			~CDeviceImage();

			_CDeviceImage AsDeviceObject() const
			{
				return *this;
			}

			void Create(const Clu::_SImageFormat& xFormat);
			void Destroy();

			void CopyFrom(const Clu::CIImage& xImage);
			void CopyInto(Clu::CIImage& xImage) const;
		};
	} // namespace Cuda
} // namespace Clu

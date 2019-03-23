////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceSurface.h
//
// summary:   Declares the device surface class
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
#include "DeviceArray2D.h"
#include "CluTec.Types1/ImageFormat.h"
#include "CluTec.Types1/IImage.h"

#include "PixelTypeInfo.h"
#include "DeviceImage.h"

namespace Clu
{
	namespace Cuda
	{
		class _CDeviceSurfMipMap;
		class CDeviceSurfMipMap;
		class CDeviceSurface;

		class _CDeviceSurface
		{
		public:
			friend class _CDeviceSurfMipMap;
			friend class CDeviceSurfMipMap;
			friend class CDeviceSurface;

			using TDeviceObject = _CDeviceSurface;

		protected:
			cudaSurfaceObject_t m_xSurface;
			Clu::_SImageFormat m_xFormat;
		public:
			int m_iOrigX, m_iOrigY;

		protected:
			void _Init()
			{
				m_xSurface = 0;
				m_xFormat.Clear();
				m_iOrigX = 0;
				m_iOrigY = 0;
			}

			__CUDA_HDI__ void _Set(cudaSurfaceObject_t xSurf, const Clu::_SImageFormat xF, int iOrigX, int iOrigY)
			{
				m_xSurface = xSurf;
				m_xFormat = xF;
				m_iOrigX = iOrigX;
				m_iOrigY = iOrigY;
			}

		public:

			__CUDA_HDI__ bool IsValid() const
			{
				return m_xSurface != 0 && m_xFormat.IsValid()
					&& m_iOrigX >= 0 && m_iOrigX < m_xFormat.iWidth
					&& m_iOrigY >= 0 && m_iOrigY < m_xFormat.iHeight;
			}

			__CUDA_HDI__ operator cudaSurfaceObject_t()
			{
				return m_xSurface;
			}

			__CUDA_HDI__ const Clu::_SImageFormat& Format() const
			{
				return m_xFormat;
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


			template<typename TElement>
			__CUDA_DI__ TElement Read2D(int iX, int iY) const
			{
				TElement xValue;
				surf2Dread(&xValue, m_xSurface, (m_iOrigX + iX) * sizeof(TElement), (m_iOrigY + iY));
				return xValue;
			}

			template<typename TPixel>
			__CUDA_DI__ TPixel ReadPixel2D(int iX, int iY) const
			{
				using TElement = typename SPixelTypeInfo<TPixel>::TElement;
				TPixel xValue;
				surf2Dread((TElement*)&xValue, m_xSurface, (m_iOrigX + iX) * sizeof(TElement), (m_iOrigY + iY));
				return xValue;
			}

			template<typename TElement>
			__CUDA_DI__ void Read2D(TElement& xValue, int iX, int iY) const
			{
				surf2Dread(&xValue, m_xSurface, (m_iOrigX + iX) * sizeof(TElement), (m_iOrigY + iY));
			}

			template<typename TPixel>
			__CUDA_DI__ void ReadPixel2D(TPixel &xValue, int iX, int iY) const
			{
				using TElement = typename SPixelTypeInfo<TPixel>::TElement;
				surf2Dread((TElement*)&xValue, m_xSurface, (m_iOrigX + iX) * sizeof(TElement), (m_iOrigY + iY));
			}


			template<typename TElement>
			__CUDA_DI__ void Write2D(const TElement& xValue, int iX, int iY)
			{
				surf2Dwrite(xValue, m_xSurface, (m_iOrigX + iX) * sizeof(TElement), m_iOrigY + iY);
			}

			template<typename TPixel>
			__CUDA_DI__ void WritePixel2D(const TPixel& xValue, int iX, int iY)
			{
				using TElement = typename SPixelTypeInfo<TPixel>::TElement;
				surf2Dwrite(*((TElement*)&xValue), m_xSurface, (m_iOrigX + iX) * sizeof(TElement), m_iOrigY + iY);
			}

			_CDeviceSurface AsDeviceObject() const
			{
				return *this;
			}
		};


		class CDeviceSurface : public _CDeviceSurface
		{
		public:
			friend class CDeviceSurfMipMap;

			using TDeviceObject = _CDeviceSurface;

		protected:
			CDeviceArray2D m_xArray;

		public:
			CDeviceSurface();

			CDeviceSurface(CDeviceSurface&& xSurf);
			CDeviceSurface& operator=(CDeviceSurface&& xSurf);

			CDeviceSurface(const CDeviceSurface& xSurf) = delete;
			CDeviceSurface& operator=(const CDeviceSurface& xSurf) = delete;

			~CDeviceSurface();

			bool IsValid()
			{
				return m_xArray.IsValid() && _CDeviceSurface::IsValid();
			}

			static bool IsValidFormat(const SImageFormat& xFormat)
			{
				return CDeviceArray2D::IsValidFormat(xFormat);
			}

			static bool IsValidType(const Clu::_SImageType& xType)
			{
				return CDeviceArray2D::IsValidType(xType);
			}

			_CDeviceSurface GetView(int iX, int iY, int iW, int iH);

			_CDeviceSurface AsDeviceObject() const
			{
				return *this;
			}

			void Create(const Clu::SImageFormat& xFormat);
			void Destroy();

			void CopyFrom(const Clu::CIImage& xImage);
			void CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY);
			void CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyFrom(const Clu::Cuda::CDeviceImage& xImage);
			void CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY);
			void CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyInto(Clu::CIImage& xImage);
			void CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY);
			void CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);
		};


	}
}

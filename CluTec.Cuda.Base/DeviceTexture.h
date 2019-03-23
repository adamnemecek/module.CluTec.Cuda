////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceTexture.h
//
// summary:   Declares the device texture class
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
#include "TextureType.h"
#include "DeviceImage.h"
#include "PixelTypeInfo.h"

namespace Clu
{
	namespace Cuda
	{
		class _CDeviceTexture
		{
		public:
			cudaTextureObject_t m_xTexture;
			Clu::_SImageFormat m_xFormat;

		protected:
			void _Init()
			{
				m_xTexture = 0;
				m_xFormat.Clear();
			}

		public:
			__CUDA_HDI__ bool IsValid() const
			{
				return m_xTexture != 0 && m_xFormat.IsValid();
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


			__CUDA_HDI__ operator cudaTextureObject_t() const
			{
				return m_xTexture;
			}


			template<typename TPixel>
			__CUDA_DI__ typename SPixelTypeInfo<TPixel>::TNormalized TexNorm2D(float fX, float fY) const
			{
				return tex2D<typename SPixelTypeInfo<TPixel>::TNormalized>(m_xTexture, fX, fY);
			}

			template<typename TPixel>
			__CUDA_DI__ typename SPixelTypeInfo<TPixel>::TElement TexRaw2D(float fX, float fY)
			{
				return tex2D<typename SPixelTypeInfo<TPixel>::TElement>(m_xTexture, fX, fY);
			}

			template<typename TPixel>
			__CUDA_DI__ typename SPixelTypeInfo<TPixel>::TNormalized TexNorm2D(float2 vecPos) const
			{
				return tex2D<typename SPixelTypeInfo<TPixel>::TNormalized>(m_xTexture, vecPos.x, vecPos.y);
			}

			template<typename TPixel>
			__CUDA_DI__ typename SPixelTypeInfo<TPixel>::TElement TexRaw2D(float2 vecPos)
			{
				return tex2D<typename SPixelTypeInfo<TPixel>::TElement>(m_xTexture, vecPos.x, vecPos.y);
			}

		};


		class CDeviceTexture : public _CDeviceTexture
		{
		private:
			CDeviceArray2D m_xArray;
			STextureType m_xType;

		protected:
			void _Init();

		public:
			CDeviceTexture();

			CDeviceTexture(CDeviceTexture&& xTex);
			CDeviceTexture& operator=(CDeviceTexture&& xTex);

			CDeviceTexture(const CDeviceTexture& xTex) = delete;
			CDeviceTexture& operator=(const CDeviceTexture& xTex) = delete;

			~CDeviceTexture();

			bool IsValid()
			{
				return m_xArray.IsValid() && _CDeviceTexture::IsValid() && m_xFormat.IsValid();
			}

			const STextureType& Type()
			{
				return m_xType;
			}

			void Create(const Clu::SImageFormat& xFormat, const STextureType& xType);

			void Destroy();

			void CopyFrom(const Clu::CIImage& xImage);
			void CopyFrom(const CDeviceImage& xImage);
		};


	}
}

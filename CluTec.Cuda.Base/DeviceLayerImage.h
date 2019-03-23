////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceLayerImage.h
//
// summary:   Declares the device layer image class
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
#include "CluTec.Types1/ILayerImage.h"
#include "CluTec.Base/Exception.h"

#include "PixelTypeInfo.h"

#include "Api.h"
#include "DeviceArray1D.h"


namespace Clu
{
	namespace Cuda
	{
		class _CDeviceLayerImage : protected _CDeviceArray1D<uint8_t>
		{
		public:
			using TDeviceObject = _CDeviceLayerImage;

		protected:
			Clu::_SImageFormat m_xFormat;
			size_t m_nLayerPixelCount;

		public:
			__CUDA_HDI__ const Clu::_SImageFormat& Format() const
			{
				return m_xFormat;
			}

			__CUDA_HDI__ size_t LayerPixelCount() const
			{
				return m_nLayerPixelCount;
			}

			__CUDA_HDI__ size_t LayerOffset(int iLayer) const
			{
				return size_t(iLayer) * LayerPixelCount();
			}

			__CUDA_HDI__ size_t LayerPixelOffset(int iX, int iY) const
			{
				return m_xFormat.GetPixelIndex(iX, iY);
			}

			__CUDA_HDI__ size_t GetLayerPixelIndex(int iX, int iY, int iLayer) const
			{
				return LayerOffset(iLayer) + LayerPixelOffset(iX, iY);
			}

			template<typename TData>
			__CUDA_HDI__ TData& At(int iX, int iY, int iLayer)
			{
				return ((TData *)m_pData)[GetLayerPixelIndex(iX, iY, iLayer)];
			}

			template<typename TData>
			__CUDA_HDI__ const TData& At(int iX, int iY, int iLayer) const
			{
				return ((TData *)m_pData)[GetLayerPixelIndex(iX, iY, iLayer)];
			}

			template<typename TData>
			__CUDA_HDI__ TData& At(size_t nIdx)
			{
				return ((TData *)m_pData)[nIdx];
			}

			__CUDA_HDI__ void* DataPointer(int iLayer)
			{
				return (void*)&m_pData[LayerOffset(iLayer)];
			}

			__CUDA_HDI__ const void* DataPointer(int iLayer) const
			{
				return (void*)&m_pData[LayerOffset(iLayer)];
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

			// ///////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////

#pragma region Read Pixel
			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_Lum) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.r() = ((TData*)m_pData)[nPixelOffset];
			}

			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_LumA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.r() = ((TData*)m_pData)[nPixelOffset];
				pixData.a() = ((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount];
			}

			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_RGB) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.r() = ((TData*)m_pData)[nPixelOffset];
				pixData.g() = ((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount];
				pixData.b() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)];
			}

			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_RGBA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.r() = ((TData*)m_pData)[nPixelOffset];
				pixData.g() = ((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount];
				pixData.b() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)];
				pixData.a() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1) + m_nLayerPixelCount];
			}

			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_BGR) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.b() = ((TData*)m_pData)[nPixelOffset];
				pixData.g() = ((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount];
				pixData.r() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)];
			}

			template<typename TPixel>
			__CUDA_HDI__ void _ReadPixel2D(TPixel& pixData, int iX, int iY, T_BGRA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				pixData.b() = ((TData*)m_pData)[nPixelOffset];
				pixData.g() = ((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount];
				pixData.r() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)];
				pixData.a() = ((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1) + m_nLayerPixelCount];
			}

			template<typename TPixel>
			__CUDA_DI__ TPixel ReadPixel2D(int iX, int iY) const
			{
				TPixel pixData;
				ReadPixel2D(pixData, iX, iY);
				return pixData;
			}

			template<typename TPixel>
			__CUDA_DI__ void ReadPixel2D(TPixel &pixData, int iX, int iY) const
			{
				_ReadPixel2D<TPixel>(pixData, iX, iY, TPixel::TPixelType());
			}

#pragma endregion Read Pixel

			// ///////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////

#pragma region Write Pixel

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_Lum) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.r();
			}

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_LumA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.r();
				((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount] = pixData.a();
			}

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_RGB) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.r();
				((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount] = pixData.g();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)] = pixData.b();
			}

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_RGBA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.r();
				((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount] = pixData.g();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)] = pixData.b();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1) + m_nLayerPixelCount] = pixData.a();
			}

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_BGR) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.b();
				((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount] = pixData.g();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)] = pixData.r();
			}

			template<typename TPixel>
			__CUDA_HDI__ void _WritePixel2D(const TPixel& pixData, int iX, int iY, T_BGRA) const
			{
				using TData = typename TPixel::TData;
				const size_t nPixelOffset = LayerPixelOffset(iX, iY);

				((TData*)m_pData)[nPixelOffset] = pixData.b();
				((TData*)m_pData)[nPixelOffset + m_nLayerPixelCount] = pixData.g();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1)] = pixData.r();
				((TData*)m_pData)[nPixelOffset + (m_nLayerPixelCount << 1) + m_nLayerPixelCount] = pixData.a();
			}

			template<typename TPixel>
			__CUDA_DI__ void WritePixel2D(const TPixel &pixData, int iX, int iY) const
			{
				_WritePixel2D<TPixel>(pixData, iX, iY, TPixel::TPixelType());
			}

#pragma endregion Write Pixel
			_CDeviceLayerImage AsDeviceObject() const
			{
				return *this;
			}
		};


		class CDeviceLayerImage : public _CDeviceLayerImage
		{
		public:
			using TDeviceObject = _CDeviceLayerImage;

		public:
			CDeviceLayerImage();
			CDeviceLayerImage(const Clu::_SImageFormat& xFormat);

			~CDeviceLayerImage();

			_CDeviceLayerImage AsDeviceObject() const
			{
				return *this;
			}

			void Create(const Clu::_SImageFormat& xFormat);
			void Destroy();

			void CopyFrom(const Clu::CILayerImage& xImage);

			void CopyInto(Clu::CILayerImage& xImage) const;
		};
	} // namespace Cuda
} // namespace Clu

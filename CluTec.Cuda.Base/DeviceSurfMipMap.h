////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceSurfMipMap.h
//
// summary:   Declares the device surf mip map class
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
#include "DeviceSurface.h"

namespace Clu
{
	namespace Cuda
	{
		class _CDeviceSurfMipMap
		{
		protected:
			cudaSurfaceObject_t m_xSurface;

			/// <summary>	The image format of the image at mip-map level 0. </summary>
			Clu::_SImageFormat m_xFormat;

			/// <summary>	Number of mip map levels. If this is 1, we only have the full resolution image. </summary>
			int m_iActiveMipMapLevelCount;

			/// <summary>	The offset in x to the start of the minified image versions. </summary>
			int m_iMipMapOffsetX;

			int m_iMaxMipMapLevelCount;

		protected:
			void _Init()
			{
				m_xSurface = 0;
				m_xFormat.Clear();
				m_iMipMapOffsetX = 0;
				m_iActiveMipMapLevelCount = 0;
				m_iMaxMipMapLevelCount = 0;
			}


		public:
			__CUDA_HDI__ bool IsValid()
			{
				return m_xSurface != 0 && m_xFormat.IsValid() && m_iActiveMipMapLevelCount > 0;
			}

			__CUDA_HDI__ operator cudaSurfaceObject_t()
			{
				return m_xSurface;
			}

			static __CUDA_HDI__ int NextLevelWidth(int iW)
			{
				return (iW - 1) / 2;
			}

			static __CUDA_HDI__ int NextLevelHeight(int iH)
			{
				return (iH - 1) / 2;
			}

			__CUDA_HDI__ Clu::_SImageFormat Format(int iLevel) const
			{
				Clu::_SImageFormat xF(m_xFormat);

				for (int i = 0; i < iLevel && i < m_iMaxMipMapLevelCount; ++i)
				{
					xF.iWidth = NextLevelWidth(xF.iWidth);
					xF.iHeight = NextLevelHeight(xF.iHeight);
				}

				return xF;
			}

			__CUDA_HDI__ void GetOrigin(int& iX, int &iY, int iLevel)
			{
				if (iLevel == 0)
				{
					iX = 0;
					iY = 0;
				}
				else
				{
					iX = m_iMipMapOffsetX;
					iY = 0;

					int iH = m_xFormat.iHeight;
					for (int i = 1; i < iLevel; ++i)
					{
						iH = NextLevelHeight(iH);
						iY += iH;
					}
				}
			}

			__CUDA_HDI__ int MipMapLevelCount()
			{
				return m_iActiveMipMapLevelCount;
			}

			__CUDA_HDI__ int MaxMipMapLevelCount()
			{
				return m_iMaxMipMapLevelCount;
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


			template<typename TElement>
			__CUDA_DI__ TElement Read2D(int iX, int iY) const
			{
				TElement xValue;

				surf2Dread(&xValue, m_xSurface, iX * sizeof(TElement), iY);
				return xValue;
			}

			template<typename TPixel>
			__CUDA_DI__ typename SPixelTypeInfo<TPixel>::TElement ReadPixel2D(int iX, int iY) const
			{
				return Read2D<typename SPixelTypeInfo<TPixel>::TElement>(iX, iY);
			}


			template<typename TElement>
			__CUDA_DI__ void Write2D(const TElement& xValue, int iX, int iY)
			{
				surf2Dwrite(xValue, m_xSurface, iX * sizeof(TElement), iY);
			}

			template<typename TPixel>
			__CUDA_DI__ void WritePixel2D(const typename SPixelTypeInfo<TPixel>::TElement& xValue, int iX, int iY)
			{
				Write2D(xValue, iX, iY);
			}

			__CUDA_HDI__ _CDeviceSurface GetMipMap(int iLevel)
			{
				_CDeviceSurface xView;

				int iX, iY;
				GetOrigin(iX, iY, iLevel);
				xView._Set(m_xSurface, Format(iLevel), iX, iY);

				return xView;
			}
		};


		class CDeviceSurfMipMap : public _CDeviceSurfMipMap
		{
		private:
			CDeviceArray2D m_xArray;

		protected:
			void _EvalMaxMipMapLevelCount();

		public:
			CDeviceSurfMipMap();

			CDeviceSurfMipMap(CDeviceSurfMipMap&& xSurf);
			CDeviceSurfMipMap& operator=(CDeviceSurfMipMap&& xSurf);

			CDeviceSurfMipMap(const CDeviceSurfMipMap& xSurf) = delete;
			CDeviceSurfMipMap& operator=(const CDeviceSurfMipMap& xSurf) = delete;

			~CDeviceSurfMipMap();

			bool IsValid()
			{
				return m_xArray.IsValid() && _CDeviceSurfMipMap::IsValid();
			}

			Clu::SImageFormat AllLevelFormat()
			{
				return m_xArray.ActiveFormat();
			}

			_CDeviceSurface GetView();

			void SetActiveMipMapLevelCount(int iCount);
			int GetActiveMipMapLevelCount();

			void Create(const Clu::SImageFormat& xFormat);
			void Destroy();

			void CopyFrom(const Clu::CIImage& xImage, int iLevel);
			void CopyFrom(const Clu::Cuda::CDeviceImage& xImage, int iLevel);

			void CopyInto(Clu::CIImage& xImage, int iLevel);
			void CopyInto(Clu::CIImage& xImage, int iLevel, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyInto(Clu::CIImage& xImage);
			void CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyInto(Clu::Cuda::CDeviceSurface& xImage, int iLevel);
		};


	}
}

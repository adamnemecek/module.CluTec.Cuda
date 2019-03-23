////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceSurfMipMap.cpp
//
// summary:   Implements the device surf mip map class
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

#include "CluTec.Base\Logger.h"

#include "DeviceSurfMipMap.h"

namespace Clu
{
	namespace Cuda
	{


		CDeviceSurfMipMap::CDeviceSurfMipMap()
		{
			_Init();
		}

		CDeviceSurfMipMap::CDeviceSurfMipMap(CDeviceSurfMipMap&& xSurf)
		{
			*this = std::forward<CDeviceSurfMipMap>(xSurf);
		}

		CDeviceSurfMipMap& CDeviceSurfMipMap::operator = (CDeviceSurfMipMap&& xSurf)
		{
			m_xArray = std::move(xSurf.m_xArray);

			m_xSurface = xSurf.m_xSurface;
			m_xFormat = xSurf.m_xFormat;

			xSurf._Init();
			return *this;
		}

		CDeviceSurfMipMap::~CDeviceSurfMipMap()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceSurfMipMap)
		}


		void CDeviceSurfMipMap::_EvalMaxMipMapLevelCount()
		{
			int iW = m_xFormat.iWidth;
			int iH = m_xFormat.iHeight;

			m_iMaxMipMapLevelCount = 0;
			iW = NextLevelWidth(iW);
			iH = NextLevelHeight(iH);

			while (iW > 0 && iH > 0)
			{
				++m_iMaxMipMapLevelCount;

				iW = NextLevelWidth(iW);
				iH = NextLevelHeight(iH);
			}
		}


		void CDeviceSurfMipMap::SetActiveMipMapLevelCount(int iCount)
		{
			if (iCount < 0 || iCount > m_iMaxMipMapLevelCount)
			{
				throw CLU_EXCEPTION("Given mip-map level count out of range");
			}

			m_iActiveMipMapLevelCount = iCount;
		}
		
		
		int CDeviceSurfMipMap::GetActiveMipMapLevelCount()
		{
			return m_iActiveMipMapLevelCount;
		}


		_CDeviceSurface CDeviceSurfMipMap::GetView()
		{
			_CDeviceSurface xSurf;

			xSurf._Set(m_xSurface, m_xArray.ActiveFormat(), 0, 0);
			return xSurf;
		}

		void CDeviceSurfMipMap::Create(const Clu::SImageFormat& xFormat)
		{
			try
			{
				if (xFormat.DimOf(xFormat.ePixelType) == 3)
				{
					throw CLU_EXCEPTION("Channel count of pixel type has to be 1, 2 or 4");
				}

				if (IsValid() && xFormat == m_xFormat)
				{
					return;
				}

				Clu::SImageFormat xF(xFormat);

				// Ensure that the MipMap images start at a 4 pixel boundary, which is what
				// makes the pixel interleaving of the surface easiest.
				m_iMipMapOffsetX = (((xF.iWidth / 4) + (xF.iWidth % 4 > 0 ? 1 : 0)) * 4);

				xF.iWidth = m_iMipMapOffsetX + NextLevelWidth(xF.iWidth) - 1 + 1;
				
				if (m_xArray.Create(xF, EDeviceArrayAllocation::SurfaceLoadStore))
				{
					if (m_xSurface)
					{
						Cuda::DestroySurfaceObject(m_xSurface);
					}

					Cuda::CreateSurfaceObject(&m_xSurface, m_xArray);
				}

				m_xFormat = xFormat;
				m_iActiveMipMapLevelCount = 1;
				_EvalMaxMipMapLevelCount();
			}
			CLU_CATCH_RETHROW_ALL("Error creating CUDA mip-map surface")

		}

		void CDeviceSurfMipMap::Destroy()
		{
			try
			{
				Cuda::DestroySurfaceObject(m_xSurface);
				m_xArray.Destroy();
				_Init();
			}
			CLU_CATCH_RETHROW_ALL("Error destroying CUDA mip-map surface")
		}


		void CDeviceSurfMipMap::CopyFrom(const Clu::CIImage& xImage, int iLevel)
		{
			try
			{
				if (iLevel < 0 || iLevel >= m_iActiveMipMapLevelCount)
				{
					throw CLU_EXCEPTION("Invalid mip-map level");
				}

				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				Clu::SImageFormat xLevelF = Format(iLevel);

				if (!IsValid() || xLevelF != xImage.Format())
				{
					throw CLU_EXCEPTION("Given image does not have the correct format for the specified mip-map level");
				}

				int iTrgX, iTrgY;
				GetOrigin(iTrgX, iTrgY, iLevel);

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY, 0, 0, xLevelF.iWidth, xLevelF.iHeight);
			}
			CLU_CATCH_RETHROW_ALL("Error copying to CUDA mip-map surface")
		}

		void CDeviceSurfMipMap::CopyFrom(const Clu::Cuda::CDeviceImage& xImage, int iLevel)
		{
			try
			{
				if (iLevel < 0 || iLevel >= m_iActiveMipMapLevelCount)
				{
					throw CLU_EXCEPTION("Invalid mip-map level");
				}

				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				Clu::SImageFormat xLevelF = Format(iLevel);

				if (!IsValid() || xLevelF != xImage.Format())
				{
					throw CLU_EXCEPTION("Given image does not have the correct format for the specified mip-map level");
				}

				int iTrgX, iTrgY;
				GetOrigin(iTrgX, iTrgY, iLevel);

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY, 0, 0, xLevelF.iWidth, xLevelF.iHeight);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA mip-map surface")
		}


		void CDeviceSurfMipMap::CopyInto(Clu::CIImage& xImage, int iLevel)
		{
			SImageFormat xF = Format(iLevel);
			CopyInto(xImage, iLevel, 0, 0, 0, 0, xF.iWidth, xF.iHeight);
		}

		void CDeviceSurfMipMap::CopyInto(Clu::CIImage& xImage, int iLevel, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (iLevel < 0 || iLevel >= m_iActiveMipMapLevelCount)
				{
					throw CLU_EXCEPTION("Invalid mip-map level");
				}

				if (!IsValid())
				{
					throw CLU_EXCEPTION("Surface is invalid");
				}

				Clu::SImageFormat xF = Format(iLevel);

				if (!xImage.IsValid())
				{
					xImage.Create(xF);
				}
				else
				{
					if (xImage.Format() != xF)
					{
						if (xImage.IsDataOwner())
						{
							xImage.Create(xF);
						}
						else
						{
							throw CLU_EXCEPTION("Format of given image does not match device image");
						}
					}
				}

				int iOrigX, iOrigY;
				GetOrigin(iOrigX, iOrigY, iLevel);

				m_xArray.CopyInto(xImage, iTrgX, iTrgY, iOrigX + iSrcX, iOrigY + iSrcY, iSrcW, iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA mip-map surface to host memory")
		}



		void CDeviceSurfMipMap::CopyInto(Clu::CIImage& xImage)
		{
			Clu::SImageFormat xF = AllLevelFormat();
			CopyInto(xImage, 0, 0, 0, 0, xF.iWidth, xF.iHeight);
		}


		void CDeviceSurfMipMap::CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Surface is invalid");
				}

				Clu::SImageFormat xF = AllLevelFormat();

				if (!xImage.IsValid())
				{
					xImage.Create(xF);
				}
				else
				{
					if (xImage.Format() != xF)
					{
						if (xImage.IsDataOwner())
						{
							xImage.Create(xF);
						}
						else
						{
							throw CLU_EXCEPTION("Format of given image does not match device image");
						}
					}
				}

				m_xArray.CopyInto(xImage, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA mip-map surface to host memory")
		}







		void CDeviceSurfMipMap::CopyInto(Clu::Cuda::CDeviceSurface& xImage, int iLevel)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Mip-map is invalid");
				}

				Clu::SImageFormat xF(Format(iLevel));

				if (!xImage.IsValid() || xF != xImage.Format())
				{
					xImage.Create(xF);
				}

				int iSrcX, iSrcY;
				GetOrigin(iSrcX, iSrcY, iLevel);

				xImage.m_xArray.CopyFrom(m_xArray, 0, 0, iSrcX, iSrcY, xF.iWidth, xF.iHeight);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA mip-map surface to surface")
		}


	}
}

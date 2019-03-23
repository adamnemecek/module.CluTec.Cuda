////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceSurface.cpp
//
// summary:   Implements the device surface class
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

#include "DeviceSurface.h"

namespace Clu
{
	namespace Cuda
	{


		CDeviceSurface::CDeviceSurface()
		{
			_Init();
		}

		CDeviceSurface::CDeviceSurface(CDeviceSurface&& xSurf)
		{
			*this = std::forward<CDeviceSurface>(xSurf);
		}

		CDeviceSurface& CDeviceSurface::operator = (CDeviceSurface&& xSurf)
		{
			m_xArray = std::move(xSurf.m_xArray);

			m_xSurface = xSurf.m_xSurface;
			m_xFormat = xSurf.m_xFormat;

			xSurf._Init();
			return *this;
		}

		CDeviceSurface::~CDeviceSurface()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceSurface)
		}


		_CDeviceSurface CDeviceSurface::GetView(int iX, int iY, int iW, int iH)
		{

			if (iX < 0 || iW < 0 || iY < 0 || iH < 0
				|| iX + iW > m_xFormat.iWidth
				|| iY + iH > m_xFormat.iHeight)
			{
				throw CLU_EXCEPTION("Invalid view shape");
			}

			Clu::_SImageFormat xF(m_xFormat);
			_CDeviceSurface xSurf;
			
			xF.iWidth = iW;
			xF.iHeight = iH;

			xSurf._Set(m_xSurface, xF, iX, iY);
			return xSurf;
		}

		void CDeviceSurface::Create(const Clu::SImageFormat& xFormat)
		{
			try
			{
				if (IsValid() && xFormat == m_xFormat)
				{
					return;
				}

				// If create returns false, then no new array was created
				// and we can reuse the current one with a different format
				// descriptor.
				if (m_xArray.Create(xFormat, EDeviceArrayAllocation::SurfaceLoadStore))
				{
					// a new array was created.
					if (m_xSurface != 0)
					{
						Cuda::DestroySurfaceObject(m_xSurface);
					}
					Cuda::CreateSurfaceObject(&m_xSurface, m_xArray);
				}

				// NOTE: The active format of the array may differ from the asked for
				// format, since the width of an array has to be a multiple of 4 bytes.
				m_xFormat = m_xArray.ActiveFormat();
			}
			CLU_CATCH_RETHROW_ALL("Error creating CUDA surface")

		}

		void CDeviceSurface::Destroy()
		{
			try
			{
				Cuda::DestroySurfaceObject(m_xSurface);
				m_xArray.Destroy();
				_Init();
			}
			CLU_CATCH_RETHROW_ALL("Error destroying CUDA surface")
		}


		void CDeviceSurface::CopyFrom(const Clu::CIImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || m_xFormat != xImage.Format())
				{
					Create(xImage.Format());
				}

				m_xArray.CopyFrom(xImage);
			}
			CLU_CATCH_RETHROW_ALL("Error copying to CUDA surface")
		}

		void CDeviceSurface::CopyFrom(const Clu::Cuda::CDeviceImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				if (!IsValid() || m_xFormat != xImage.Format())
				{
					Create(xImage.Format());
				}

				m_xArray.CopyFrom(xImage);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA surface")
		}



		void CDeviceSurface::CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA surface")
		}

		void CDeviceSurface::CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA surface")
		}

		void CDeviceSurface::CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA surface")
		}

		void CDeviceSurface::CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given device image is invalid");
				}

				m_xArray.CopyFrom(xImage, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying device image to CUDA surface")
		}

		void CDeviceSurface::CopyInto(Clu::CIImage& xImage)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Surface is invalid");
				}

				if (!xImage.IsValid() || m_xFormat != xImage.Format())
				{
					xImage.Create(m_xFormat);
				}

				m_xArray.CopyInto(xImage);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA surface to host memory")
		}

		void CDeviceSurface::CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Surface is invalid");
				}

				m_xArray.CopyInto(xImage, iTrgX, iTrgY);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA surface to host memory")
		}

		void CDeviceSurface::CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Surface is invalid");
				}

				m_xArray.CopyInto(xImage, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA surface to host memory")
		}

	}
}

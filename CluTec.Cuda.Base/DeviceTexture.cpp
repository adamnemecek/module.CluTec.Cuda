////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceTexture.cpp
//
// summary:   Implements the device texture class
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

#include "CluTec.Base/Logger.h"
#include "DeviceTexture.h"

namespace Clu
{
	namespace Cuda
	{


		CDeviceTexture::CDeviceTexture()
		{
			_Init();
		}

		CDeviceTexture::CDeviceTexture(CDeviceTexture&& xTex)
		{
			*this = std::forward<CDeviceTexture>(xTex);
		}

		CDeviceTexture& CDeviceTexture::operator = (CDeviceTexture&& xTex)
		{
			m_xArray = std::move(xTex.m_xArray);

			m_xTexture = xTex.m_xTexture;
			m_xFormat = xTex.m_xFormat;

			xTex._Init();
			return *this;
		}

		CDeviceTexture::~CDeviceTexture()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceTexture)
		}

		void CDeviceTexture::_Init()
		{
			_CDeviceTexture::_Init();

			m_xType.Reset();
			m_xArray.Destroy();
		}


		void CDeviceTexture::Create(const Clu::SImageFormat& xFormat, const STextureType& xType)
		{
			try
			{
				if (IsValid() && xFormat == m_xFormat)
				{
					return;
				}


				if (m_xArray.Create(xFormat, EDeviceArrayAllocation::Default)
					|| m_xType != xType)
				{
					if (m_xTexture != 0)
					{
						Cuda::DestroyTextureObject(m_xTexture);
					}

					m_xType = xType;
					cudaTextureDesc xT = (cudaTextureDesc)m_xType;

					Cuda::CreateTextureObject(&m_xTexture, m_xArray, &xT, nullptr);
				}

				CLU_ASSERT(xFormat == m_xArray.ActiveFormat());

				m_xFormat = m_xArray.ActiveFormat();
			}
			CLU_CATCH_RETHROW_ALL("Error creating CUDA texture")

		}

		void CDeviceTexture::Destroy()
		{
			try
			{
				Cuda::DestroyTextureObject(m_xTexture);
				m_xArray.Destroy();
				_Init();
			}
			CLU_CATCH_RETHROW_ALL("Error destroying CUDA texture")
		}


		void CDeviceTexture::CopyFrom(const Clu::CIImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || m_xFormat != xImage.Format())
				{
					throw CLU_EXCEPTION("Inappropriate texture for given image");
				}

				m_xArray.CopyFrom(xImage);
			}
			CLU_CATCH_RETHROW_ALL("Error copying to CUDA texture")
		}

		void CDeviceTexture::CopyFrom(const CDeviceImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || m_xFormat != xImage.Format())
				{
					throw CLU_EXCEPTION("Inappropriate texture for given image");
				}

				m_xArray.CopyFrom(xImage);
			}
			CLU_CATCH_RETHROW_ALL("Error copying to CUDA texture")
		}



	}
}

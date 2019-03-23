////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceImage.cpp
//
// summary:   Implements the device image class
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

#include "DeviceImage.h"
#include "CluTec.Base/Logger.h"

namespace Clu
{
	namespace Cuda
	{
		CDeviceImage::CDeviceImage()
		{
			m_xFormat.Clear();
			_CDeviceArray1D::Clear();
		}

		CDeviceImage::CDeviceImage(const Clu::_SImageFormat& xFormat)
		{
			try
			{
				Create(xFormat);
			}
			CLU_CATCH_RETHROW_ALL("Error constructing device image")
		}

		CDeviceImage::~CDeviceImage()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceImage)
		}

		void CDeviceImage::Create(const Clu::_SImageFormat& xFormat)
		{
			try
			{
				// If a memory block already exists that is large enough
				// for the new image format, then there is no new allocation.
				_CDeviceArray1D::Create(xFormat.ByteCount());

				m_xFormat = xFormat;
			}
			CLU_CATCH_RETHROW_ALL("Error creating device image")
		}

		void CDeviceImage::Destroy()
		{
			try
			{
				m_xFormat.Clear();
				_CDeviceArray1D::Destroy();
			}
			CLU_CATCH_RETHROW_ALL("Error destroying device image")
		}


		void CDeviceImage::CopyFrom(const Clu::CIImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || xImage.Format() != Format())
				{
					Create(xImage.Format());
				}

				_CDeviceArray1D::ToDevice((const TValuePtr)xImage.DataPointer());
			}
			CLU_CATCH_RETHROW_ALL("Error copying image to device")
		}

		void CDeviceImage::CopyInto(Clu::CIImage& xImage) const
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Device image is not valid");
				}

				if (!xImage.IsValid())
				{
					xImage.Create(Format());
				}
				else
				{
					if (xImage.Format() != Format())
					{
						if (xImage.IsDataOwner())
						{
							xImage.Create(Format());
						}
						else
						{
							throw CLU_EXCEPTION("Format of given image does not match device image");
						}
					}
				}

				_CDeviceArray1D::ToHost((TValuePtr)xImage.DataPointer());
			}
			CLU_CATCH_RETHROW_ALL("Error copying image to host")
		}



	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceLayerImage.cpp
//
// summary:   Implements the device layer image class
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

#include "DeviceLayerImage.h"
#include "CluTec.Base/Logger.h"

namespace Clu
{
	namespace Cuda
	{
		CDeviceLayerImage::CDeviceLayerImage()
		{
			m_nLayerPixelCount = 0;
			m_xFormat.Clear();
			_CDeviceArray1D::Clear();
		}

		CDeviceLayerImage::CDeviceLayerImage(const Clu::_SImageFormat& xFormat)
		{
			try
			{
				Create(xFormat);
			}
			CLU_CATCH_RETHROW_ALL("Error constructing device image")
		}

		CDeviceLayerImage::~CDeviceLayerImage()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceLayerImage)
		}

		void CDeviceLayerImage::Create(const Clu::_SImageFormat& xFormat)
		{
			try
			{
				// If a memory block already exists that is large enough
				// for the new image format, then there is no new allocation.
				_CDeviceArray1D::Create(xFormat.ByteCount());

				m_xFormat = xFormat;
				m_nLayerPixelCount = m_xFormat.PixelCount();
			}
			CLU_CATCH_RETHROW_ALL("Error creating device image")
		}

		void CDeviceLayerImage::Destroy()
		{
			try
			{
				m_nLayerPixelCount = 0;
				m_xFormat.Clear();
				_CDeviceArray1D::Destroy();
			}
			CLU_CATCH_RETHROW_ALL("Error destroying device image")
		}


		void CDeviceLayerImage::CopyFrom(const Clu::CILayerImage& xImage)
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

				if (xImage.IsCompactMemoryBlock())
				{
					_CDeviceArray1D::ToDevice((const TValuePtr)xImage.DataPointer(0));
				}
				else
				{
					size_t nByteOffset = 0;
					const size_t nByteCount = xImage.LayerByteCount();

					const int iLayerCount = (int)xImage.LayerCount();
					for (int iLayerIdx = 0; iLayerIdx < iLayerCount; ++iLayerIdx)
					{
						_CDeviceArray1D::ToDevice((const TValuePtr)xImage.DataPointer(iLayerIdx), nByteOffset, nByteCount);
						nByteOffset += nByteCount;
					}
				}
			}
			CLU_CATCH_RETHROW_ALL("Error copying image to device")
		}


		void CDeviceLayerImage::CopyInto(Clu::CILayerImage& xImage) const
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

				if (xImage.IsCompactMemoryBlock())
				{
					_CDeviceArray1D::ToHost((TValuePtr)xImage.DataPointer(0));
				}
				else
				{
					size_t nByteOffset = 0;
					const size_t nByteCount = xImage.LayerByteCount();

					const int iLayerCount = (int)xImage.LayerCount();
					for (int iLayerIdx = 0; iLayerIdx < iLayerCount; ++iLayerIdx)
					{
						_CDeviceArray1D::ToHost((const TValuePtr)xImage.DataPointer(iLayerIdx), nByteOffset, nByteCount);
						nByteOffset += nByteCount;
					}
				}
			}
			CLU_CATCH_RETHROW_ALL("Error copying image to host")
		}

	}
}
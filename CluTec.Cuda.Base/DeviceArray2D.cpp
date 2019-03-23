////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceArray2D.cpp
//
// summary:   Implements the device array 2D class
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

#include "CluTec.Base/Exception.h"
#include "CluTec.Base/Logger.h"

#include "DeviceArray2D.h"


namespace Clu
{
	namespace Cuda
	{


		CDeviceArray2D::CDeviceArray2D()
		{
			_Init();
		}

		CDeviceArray2D::CDeviceArray2D(CDeviceArray2D&& xDevArray)
		{
			*this = std::forward<CDeviceArray2D>(xDevArray);
		}

		CDeviceArray2D& CDeviceArray2D::operator= (CDeviceArray2D&& xDevArray)
		{
			m_xChannelFormat = xDevArray.m_xChannelFormat;
			m_xActiveFormat = xDevArray.m_xActiveFormat;
			m_xCapacityFormat = xDevArray.m_xCapacityFormat;
			m_xResource = xDevArray.m_xResource;
			m_pArray = xDevArray.m_pArray;

			xDevArray._Init();

			return *this;
		}

		CDeviceArray2D::~CDeviceArray2D()
		{
			try
			{
				Destroy();
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceArray2D)
		}

		void CDeviceArray2D::_Init()
		{
			m_xChannelFormat.Clear();
			m_xActiveFormat.Clear();
			m_xCapacityFormat.Clear();

			memset(&m_xResource, 0, sizeof(cudaResourceDesc));
			m_xResource.resType = cudaResourceTypeArray;

			m_pArray = nullptr;
		}

		bool CDeviceArray2D::IsValid() const
		{
			return m_pArray != nullptr;
		}

		bool CDeviceArray2D::IsValidFormat(const Clu::SImageFormat& xFormat)
		{
			return IsValidType(xFormat);
		}

		bool CDeviceArray2D::IsValidType(const Clu::_SImageType& xType)
		{
			int iChannelCount = (int)Clu::SImageFormat::DimOf(xType.ePixelType);
			return (iChannelCount == 1 || iChannelCount == 2 || iChannelCount == 4);
		}

		void CDeviceArray2D::Destroy()
		{
			try
			{
				if (IsValid())
				{
					Cuda::FreeArray(m_pArray);
					_Init();
				}
			}
			CLU_CATCH_RETHROW_ALL("Error destroying 2D CUDA array")
		}

		void CDeviceArray2D::_AdjustFormatSize(Clu::SImageFormat& xFormat)
		{
			const int iBytesPerPixel = (int)xFormat.BytesPerPixel();
			const int iWidthInBytes = xFormat.iWidth * iBytesPerPixel;
			const int iWidthInWords = iWidthInBytes / 4 + (iWidthInBytes % 4 > 0 ? 1 : 0);
			xFormat.iWidth = (iWidthInWords * 4) / iBytesPerPixel;
		}

		bool CDeviceArray2D::Create(const Clu::SImageFormat& xImageFormat, EDeviceArrayAllocation eAlloc)
		{
			try
			{
				bool bNewArray = false;

				if (!xImageFormat.IsValid())
				{
					throw CLU_EXCEPTION("Invalid image format given");
				}

				if (!IsValidFormat(xImageFormat))
				{
					throw CLU_EXCEPTION("Can only create a 2D Cuda array with 1, 2 or 4 channels.");
				}

				Clu::SImageFormat xF(xImageFormat);
				// Ensure that the width is an integer multiple of words.
				_AdjustFormatSize(xF);

				if (!IsValid() 
					|| !m_xCapacityFormat.IsEqualType(xF)
					|| m_xCapacityFormat.iWidth < xF.iWidth
					|| m_xCapacityFormat.iHeight < xF.iHeight)
				{
					Destroy();

					m_xChannelFormat.Set(xF);
					m_xCapacityFormat = xF;

					Cuda::MallocArray(&m_pArray, m_xChannelFormat, size_t(m_xCapacityFormat.iWidth), size_t(m_xCapacityFormat.iHeight), eAlloc);

					//cudaChannelFormatDesc xDesc;
					//cudaExtent xExt;
					//unsigned uFlags = 0;
					//cudaArrayGetInfo(&xDesc, &xExt, &uFlags, m_pArray);

					m_xResource.res.array.array = m_pArray;

					m_xActiveFormat = m_xCapacityFormat;
					bNewArray = true;
				}
				else
				{
					m_xActiveFormat = xF;
				}

				return bNewArray;
			}
			CLU_CATCH_RETHROW_ALL("Error creating 2D CUDA array")
		}


		void CDeviceArray2D::_EnsureCopyRangeOK(const Clu::SImageFormat& xTrgF, const Clu::SImageFormat& xSrcF, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			if (iTrgX < 0)
			{
				throw CLU_EXCEPTION("Negative target x-coordinate not allowed");
			}

			if (iTrgY < 0)
			{
				throw CLU_EXCEPTION("Negative target y-coordinate not allowed");
			}

			if (iSrcX < 0)
			{
				throw CLU_EXCEPTION("Negative source x-coordinate not allowed");
			}

			if (iSrcY < 0)
			{
				throw CLU_EXCEPTION("Negative source y-coordinate not allowed");
			}

			if (iSrcW < 0)
			{
				throw CLU_EXCEPTION("Negative source width not allowed");
			}

			if (iSrcH < 0)
			{
				throw CLU_EXCEPTION("Negative source height not allowed");
			}

			if (iTrgX + iSrcW > xTrgF.iWidth)
			{
				throw CLU_EXCEPTION("Given target x-coordinate and source width exceed target width");
			}

			if (iTrgY + iSrcH > xTrgF.iHeight)
			{
				throw CLU_EXCEPTION("Given target y-coordinate and source height exceed target height");
			}

			if (iSrcX + iSrcW > xSrcF.iWidth)
			{
				throw CLU_EXCEPTION("Given source x-coordinate and source width exceed source image total width");
			}

			if (iSrcY + iSrcH > xSrcF.iHeight)
			{
				throw CLU_EXCEPTION("Given source y-coordinate and source height exceed source image total height");
			}

		}



		void CDeviceArray2D::CopyFrom(const Clu::CIImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid())
				{
					throw CLU_EXCEPTION("Array is incompatible with given image format");
				}

				if (m_xActiveFormat.IsEqualSize(xImage.Format()))
				{
					// if the source image is of exactly the same size as the surface, then we can do a simple copy.
					Cuda::MemCpyToArray(m_pArray, 0, 0, xImage.DataPointer(), xImage.ByteCount(), ECopyType::HostToDevice);
				}
				else
				{
					// otherwise try to insert the source image into the surface at position (0,0)
					CopyFrom(xImage, 0, 0);
				}
			}
			CLU_CATCH_RETHROW_ALL("Error copying data to CUDA device 2D array")
		}


		void CDeviceArray2D::CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || !m_xActiveFormat.IsEqualType(xImage.Format()))
				{
					throw CLU_EXCEPTION("Array is incompatible with given image type");
				}

				_EnsureCopyRangeOK(m_xActiveFormat, xImage.Format(), iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);

				const uint8_t *pData = (const uint8_t *)xImage.DataPointer();
				size_t nPitch = size_t(xImage.Width()) * xImage.BytesPerPixel();

				pData += size_t(iSrcY) * nPitch + size_t(iSrcX) * xImage.BytesPerPixel();

				Cuda::MemCpy2DToArray(m_pArray, (size_t) iTrgX, (size_t) iTrgY
					, pData
					, nPitch, size_t(iSrcW) * xImage.BytesPerPixel(), size_t(iSrcH)
					, ECopyType::HostToDevice);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data to CUDA device 2D array")
		}

		void CDeviceArray2D::CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY)
		{
			CopyFrom(xImage, iTrgX, iTrgY, 0, 0, xImage.Width(), xImage.Height());
		}





		void CDeviceArray2D::CopyFrom(const CDeviceImage& xImage)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || !m_xActiveFormat.IsEqualType(xImage.Format()))
				{
					throw CLU_EXCEPTION("Array is incompatible with given image format");
				}

				if (m_xActiveFormat.IsEqualSize(xImage.Format()))
				{
					// if the source image is of exactly the same size as the surface, then we can do a simple copy.
					Cuda::MemCpyToArray(m_pArray, 0, 0, xImage.DataPointer(), xImage.ByteCount(), ECopyType::DeviceToDevice);
				}
				else
				{
					// otherwise try to insert the source image into the surface at position (0,0)
					CopyFrom(xImage, 0, 0);
				}
			}
			CLU_CATCH_RETHROW_ALL("Error copying data to CUDA device 2D array")
		}


		void CDeviceArray2D::CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || !m_xActiveFormat.IsEqualType(xImage.Format()))
				{
					throw CLU_EXCEPTION("Array is incompatible with given image type");
				}

				Clu::SImageFormat xSrcF(xImage.Format());
				_EnsureCopyRangeOK(m_xActiveFormat, xSrcF, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);

				const uint8_t *pData = (const uint8_t *)xImage.DataPointer();
				size_t nPitch = size_t(xSrcF.iWidth) * xSrcF.BytesPerPixel();

				pData += size_t(iSrcY) * nPitch + size_t(iSrcX) * xSrcF.BytesPerPixel();

				Cuda::MemCpy2DToArray(m_pArray
					, (size_t)iTrgX, (size_t)iTrgY
					, pData
					, nPitch
					, size_t(iSrcW) * xSrcF.BytesPerPixel(), size_t(iSrcH)
					, ECopyType::DeviceToDevice);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data to CUDA device 2D array")
		}

		void CDeviceArray2D::CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY)
		{
			CopyFrom(xImage, iTrgX, iTrgY, 0, 0, xImage.Format().iWidth, xImage.Format().iHeight);
		}




		void CDeviceArray2D::CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!xImage.IsValid())
				{
					throw CLU_EXCEPTION("Given image is invalid");
				}

				if (!IsValid() || !m_xActiveFormat.IsEqualType(xImage.Format()))
				{
					throw CLU_EXCEPTION("Array is incompatible with given image type");
				}

				Clu::SImageFormat xTrgF(xImage.Format());
				_EnsureCopyRangeOK(xTrgF, m_xActiveFormat, iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);

				uint8_t *pData = (uint8_t *)xImage.DataPointer();
				size_t nPitch = size_t(xTrgF.iWidth) * xTrgF.BytesPerPixel();

				pData += size_t(iTrgY) * nPitch + size_t(iTrgX) * xTrgF.BytesPerPixel();

				Cuda::MemCpy2DFromArray(pData, nPitch, m_pArray 
					, size_t(iSrcX) * xTrgF.BytesPerPixel(), size_t(iSrcY)
					, size_t(iSrcW) * xTrgF.BytesPerPixel(), size_t(iSrcH)
					, ECopyType::DeviceToHost);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA device 2D array to host memory")
		}


		void CDeviceArray2D::CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY)
		{
			CopyInto(xImage, iTrgX, iTrgY, 0, 0, m_xActiveFormat.iWidth, m_xActiveFormat.iHeight);
		}

		void CDeviceArray2D::CopyInto(Clu::CIImage& xImage)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Array is invalid");
				}

				if (!xImage.IsValid())
				{
					xImage.Create(m_xActiveFormat);
				}
				else
				{
					if (xImage.Format() != m_xActiveFormat)
					{
						if (xImage.IsDataOwner())
						{
							xImage.Create(m_xActiveFormat);
						}
						else
						{
							throw CLU_EXCEPTION("Format of given image does not match device image");
						}
					}
				}

				CopyInto(xImage, 0, 0, 0, 0, m_xActiveFormat.iWidth, m_xActiveFormat.iHeight);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA device 2D array to host memory")
		}



		void CDeviceArray2D::CopyFrom(const CDeviceArray2D& xArray, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH)
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Array is invalid");
				}

				if (!xArray.IsValid())
				{
					throw CLU_EXCEPTION("Given array is invalid");
				}

				if (!m_xActiveFormat.IsEqualType(xArray.ActiveFormat()))
				{
					throw CLU_EXCEPTION("Arrays do not have the same pixel and data types");
				}

				_EnsureCopyRangeOK(m_xActiveFormat, xArray.ActiveFormat(), iTrgX, iTrgY, iSrcX, iSrcY, iSrcW, iSrcH);

				Cuda::MemCpy2DArrayToArray(
					m_pArray, (size_t)iTrgX, (size_t)iTrgY
					, xArray.m_pArray, (size_t)iSrcX, (size_t)iSrcY
					, (size_t)iSrcW * m_xActiveFormat.BytesPerPixel(), (size_t)iSrcH);
			}
			CLU_CATCH_RETHROW_ALL("Error copying data from CUDA device 2D array to 2D array")

		}


	}
}

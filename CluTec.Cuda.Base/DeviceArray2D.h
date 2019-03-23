////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceArray2D.h
//
// summary:   Declares the device array 2D class
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

#pragma once
#include "Api.h"
#include "CluTec.Types1/ImageFormat.h"
#include "CluTec.Types1/IImage.h"

#include "Api.h"
#include "ChannelFormat.h"
#include "DeviceImage.h"

namespace Clu
{
	namespace Cuda
	{

		class _CDeviceSurface;

		class CDeviceArray2D
		{
		private:
			CChannelFormat m_xChannelFormat;
			Clu::SImageFormat m_xActiveFormat;
			Clu::SImageFormat m_xCapacityFormat;

			cudaArray_t m_pArray;
			cudaResourceDesc m_xResource;

		private:
			void _Init();
			void _EnsureCopyRangeOK(const Clu::SImageFormat& xTrgF, const Clu::SImageFormat& xSrcF, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Adjust the image format size to ensure that the width is a multiple of 4 bytes. </summary>
			///
			/// <param name="xFormat">	[in,out] The format to adjust. On return contains the adjusted format. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			void _AdjustFormatSize(Clu::SImageFormat& xFormat);

		public:
			CDeviceArray2D();
			CDeviceArray2D(const CDeviceArray2D&) = delete;
			CDeviceArray2D(CDeviceArray2D&& xDevArray);
			~CDeviceArray2D();

			CDeviceArray2D& operator= (const CDeviceArray2D& xDevArray) = delete;
			CDeviceArray2D& operator= (CDeviceArray2D&& xDevArray);

			operator cudaResourceDesc()
			{
				return m_xResource;
			}

			operator const cudaResourceDesc*()
			{
				return &m_xResource;
			}

			bool IsValid() const;

			static bool IsValidFormat(const Clu::SImageFormat& xFormat);
			static bool IsValidType(const Clu::_SImageType& xType);

			const Clu::SImageFormat& ActiveFormat() const
			{
				return m_xActiveFormat;
			}

			const Clu::SImageFormat& CapacityFormat() const
			{
				return m_xCapacityFormat;
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Creates a new device 2D array. If the given image format is of the same type
			/// 			as the currently active format and the given size is smaller or equal to the
			/// 			present size, no new array is created. Instead the present array is reused,
			/// 			even if the new size is smaller. The function return true if a new array was created
			/// 			and otherwise false. It throws exceptions if an error occurred. </summary>
			///
			/// <param name="xImageFormat">	The image format. </param>
			/// <param name="eAlloc">	   	The allocate. </param>
			///
			/// <returns>	true if a new array was created. If the current one is re-used the 
			/// 			function returns false. </returns>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			bool Create(const Clu::SImageFormat& xImageFormat, EDeviceArrayAllocation eAlloc);
			void Destroy();

			void CopyFrom(const Clu::CIImage& xImage);
			void CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY);
			void CopyFrom(const Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyFrom(const CDeviceImage& xImage);
			void CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY);
			void CopyFrom(const CDeviceImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyInto(Clu::CIImage& xImage);
			void CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY);
			void CopyInto(Clu::CIImage& xImage, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

			void CopyFrom(const CDeviceArray2D& xArray, int iTrgX, int iTrgY, int iSrcX, int iSrcY, int iSrcW, int iSrcH);

		};

	}
}

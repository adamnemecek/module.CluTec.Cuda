////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      ClearImage.cu
//
// summary:   clear image class
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

#include "ClearImage.h"
#include "CluTec.Types1/Pixel.h"

namespace Clu
{
	namespace Cuda
	{
		namespace ClearImage
		{
			namespace Kernel
			{
				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Homographies. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageOut">	The image out. </param>
				/// <param name="xImageIn"> 	The image in. </param>
				/// <param name="matHom">   	The matrix hom. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TElement>
				__global__ void ClearImage(Clu::Cuda::_CDeviceImage xImage, TElement pixClear)
				{
					const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
					const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

					if (xImage.IsInside(nSrcX, nSrcY))
					{
						xImage.At<TElement>(nSrcX, nSrcY) = pixClear;
					}
				}

				template<typename TElement>
				__global__ void ClearImage(Clu::Cuda::_CDeviceSurface xImage, TElement pixClear)
				{
					const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
					const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

					if (xImage.IsInside(nSrcX, nSrcY))
					{
						xImage.Write2D<TElement>(pixClear, nSrcX, nSrcY);
					}
				}

			} // namespace Kernel

			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// DRIVER
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////

			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				EvalThreadConfig(xDevice, xFormat, 0, 0, NumberOfRegisters);
			}




			template<typename TPixel>
			void CDriver::Process(Clu::Cuda::CDeviceImage& xImage, const TPixel& pixClear)
			{
				if (!xImage.IsOfType<TPixel>())
				{
					throw CLU_EXCEPTION("Image is not of same type as clear pixel");
				}

				using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixel>::TElement;

				TElement xClear;
				memcpy(&xClear, pixClear.pPixel, sizeof(TElement));

				Kernel::ClearImage CLU_KERNEL_CONFIG() (xImage, xClear);
			}


			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_Lum_Double&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_LumA_Double&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGB_Double&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_RGBA_Double&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGR_Double&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_Int8  &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_Int16 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_Int32 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_UInt16&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_UInt32&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_Single&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const TPixel_BGRA_Double&);



			template<typename TPixel>
			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImage, const TPixel& pixClear)
			{
				if (!xImage.IsOfType<TPixel>())
				{
					throw CLU_EXCEPTION("Image is not of same type as clear pixel");
				}

				using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixel>::TElement;

				TElement xClear;
				memcpy(&xClear, pixClear.pPixel, sizeof(TElement));

				Kernel::ClearImage CLU_KERNEL_CONFIG() (xImage, xClear);
			}



			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_Int8  &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_Int16 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_Int32 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_UInt8 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_UInt16&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_UInt32&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_Lum_Double&);

			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_Int8  &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_Int16 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_Int32 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_UInt16&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_UInt32&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_LumA_Double&);

			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_Int8  &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_Int16 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_Int32 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_UInt8 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_UInt16&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_UInt32&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGB_Double&);

			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_Int8  &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_Int16 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_Int32 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_UInt16&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_UInt32&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_RGBA_Double&);

			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_Int8  &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_Int16 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_Int32 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_UInt8 &);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_UInt16&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_UInt32&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGR_Double&);

			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_Int8  &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_Int16 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_Int32 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_UInt8 &);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_UInt16&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_UInt32&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_Single&);
			//template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const TPixel_BGRA_Double&);



		} // ImgProc
	} // Cuda
} // Clu


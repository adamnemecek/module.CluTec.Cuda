////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      CopyDeviceData.cu
//
// summary:   copy device data class
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

#include "cuda_runtime.h"

#include "CopyDeviceData.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace CopyDeviceData
		{
			namespace Kernel
			{
				struct Const
				{
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
				};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Transform image. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageTrg">	The image trg. </param>
				/// <param name="xImageSrc">	The image source. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixel>
				__global__ void Copy(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					using TData = typename TPixel::TData;

					int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageSrc.IsInside(nSrcX, nSrcY))
					{
						return;
					}

					TPixel pixData = xImageSrc.ReadPixel2D<TPixel>(nSrcX, nSrcY);
					xImageTrg.WritePixel2D<TPixel>(pixData, nSrcX, nSrcY);
				}

			}



			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				EvalThreadConfigBlockSize(xDevice, xFormat
					, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
					, 1, 1, 1, 1 // Offsets
					, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
					, NumberOfRegisters
					, false // Use also partial blocks
					);
			}


			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc)
			{
				if (!xImageTrg.IsEqualFormat(xImageSrc.Format()))
				{
					throw CLU_EXCEPTION("Input and output images do not have the same format");
				}
				
				if (xImageSrc.IsOfType<Clu::TPixel_Lum_UInt8>()
					&& xImageTrg.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					Kernel::Copy<Clu::TPixel_Lum_UInt8>
						CLU_KERNEL_CONFIG()
						(xImageTrg, xImageSrc);
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}

			}





		} // ImgProc
	} // Cuda
} // Clu


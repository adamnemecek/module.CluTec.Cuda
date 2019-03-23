////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      CensusTransform.cu
//
// summary:   census transform class
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

#include "CensusTransform.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace CensusTransform
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
				__global__ void Transform(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					using TData = typename TPixel::TData;

					int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageSrc.IsInside(nSrcX, nSrcY, 1, 1))
					{
						return;
					}

					TData uResult1, uResult2;
					uResult1 = TData(0);
					uResult2 = TData(0);


					//int iCenter, iValue1, iValue2;
					//iCenter = (int)xImageSrc.Read2D<TData>(nSrcX, nSrcY);

					//iValue1 = (int) xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY - 1);
					//iValue2 = (int) xImageSrc.Read2D<TData>(nSrcX + 0, nSrcY - 1);

					//uResult1 |= TData(0x80) * (iValue1 > iCenter + 10 || iValue1 < iCenter - 10);
					//uResult2 |= TData(0x40) * (iValue2 > iCenter + 10 || iValue2 < iCenter - 10);

					//iValue1 = (int)xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY - 1);
					//iValue2 = (int)xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY + 0);

					//uResult1 |= TData(0x20) * (iValue1 > iCenter + 10 || iValue1 < iCenter - 10);
					//uResult2 |= TData(0x10) * (iValue2 > iCenter + 10 || iValue2 < iCenter - 10);

					//iValue1 = (int)xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY + 0);
					//iValue2 = (int)xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY + 1);

					//uResult1 |= TData(0x08) * (iValue1 > iCenter + 10 || iValue1 < iCenter - 10);
					//uResult2 |= TData(0x04) * (iValue2 > iCenter + 10 || iValue2 < iCenter - 10);

					//iValue1 = (int)xImageSrc.Read2D<TData>(nSrcX + 0, nSrcY + 1);
					//iValue2 = (int)xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY + 1);

					//uResult1 |= TData(0x02) * (iValue1 > iCenter + 10 || iValue1 < iCenter - 10);
					//uResult2 |= TData(0x01) * (iValue2 > iCenter + 10 || iValue2 < iCenter - 10);

					TData uCenter;
					uCenter = (int)xImageSrc.Read2D<TData>(nSrcX, nSrcY);

					uResult1 |= TData(0x80) * (xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY - 1) > uCenter);
					uResult2 |= TData(0x40) * (xImageSrc.Read2D<TData>(nSrcX + 0, nSrcY - 1) > uCenter);
					uResult1 |= TData(0x20) * (xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY - 1) > uCenter);

					uResult2 |= TData(0x10) * (xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY + 0) > uCenter);
					uResult1 |= TData(0x08) * (xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY + 0) > uCenter);

					uResult2 |= TData(0x04) * (xImageSrc.Read2D<TData>(nSrcX - 1, nSrcY + 1) > uCenter);
					uResult1 |= TData(0x02) * (xImageSrc.Read2D<TData>(nSrcX + 0, nSrcY + 1) > uCenter);
					uResult2 |= TData(0x01) * (xImageSrc.Read2D<TData>(nSrcX + 1, nSrcY + 1) > uCenter);

					uResult1 |= uResult2;

					xImageTrg.Write2D<TData>(uResult1, nSrcX, nSrcY);
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
					Kernel::Transform<Clu::TPixel_Lum_UInt8>
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


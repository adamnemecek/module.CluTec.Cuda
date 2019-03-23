////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      MaskImage.cu
//
// summary:   mask image class
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

#include "MaskImage.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace MaskImage
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

				template<typename TPixel, typename TPixelMask>
				__global__ void MaskToBlack(Clu::Cuda::_CDeviceSurface xImageTrg
					, Clu::Cuda::_CDeviceSurface xImageSrc
					, Clu::Cuda::_CDeviceSurface xImageMask)
				{
					using TData = typename TPixel::TData;
					using TMask = typename TPixelMask::TData;

					int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageSrc.IsInside(nSrcX, nSrcY, 1, 1))
					{
						return;
					}

					TPixelMask pixMask = xImageMask.ReadPixel2D<TPixelMask>(nSrcX, nSrcY);
					TPixel pixResult;

					if (pixMask.r() != TMask(0))
					{
						pixResult = xImageSrc.ReadPixel2D<TPixel>(nSrcX, nSrcY);
					}
					else
					{
						pixResult.SetZero();
					}

					xImageTrg.WritePixel2D<TPixel>(pixResult, nSrcX, nSrcY);
				}


				template<typename TPixel, typename TPixelMask>
				__global__ void MaskAlpha(Clu::Cuda::_CDeviceSurface xImageTrg
					, Clu::Cuda::_CDeviceSurface xImageSrc
					, Clu::Cuda::_CDeviceSurface xImageMask)
				{
					using TData = typename TPixel::TData;
					using TMask = typename TPixelMask::TData;

					int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageSrc.IsInside(nSrcX, nSrcY, 1, 1))
					{
						return;
					}

					TPixelMask pixMask = xImageMask.ReadPixel2D<TPixelMask>(nSrcX, nSrcY);
					TPixel pixResult = xImageSrc.ReadPixel2D<TPixel>(nSrcX, nSrcY);;

					if (pixMask.r() == TMask(0))
					{
						pixResult.a() = TData(0);
					}

					xImageTrg.WritePixel2D<TPixel>(pixResult, nSrcX, nSrcY);
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


			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageTrg
				, const Clu::Cuda::_CDeviceSurface& xImageSrc
				, const Clu::Cuda::_CDeviceSurface& xImageMask)
			{
				if (!xImageTrg.IsEqualFormat(xImageSrc.Format()))
				{
					throw CLU_EXCEPTION("Input and output images do not have the same format");
				}

				if (!xImageTrg.IsEqualSize(xImageMask.Format()))
				{
					throw CLU_EXCEPTION("Input and mask images do not have the same size");
				}

				if (xImageSrc.IsOfType<Clu::TPixel_Lum_UInt8>()
					&& xImageMask.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					Kernel::MaskToBlack<Clu::TPixel_Lum_UInt8, Clu::TPixel_Lum_UInt8>
						CLU_KERNEL_CONFIG()
						(xImageTrg, xImageSrc, xImageMask);
				}
				else if (xImageSrc.IsOfType<Clu::TPixel_RGBA_UInt16>()
					&& xImageMask.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					Kernel::MaskToBlack<Clu::TPixel_RGBA_UInt16, Clu::TPixel_Lum_UInt8>
						CLU_KERNEL_CONFIG()
						(xImageTrg, xImageSrc, xImageMask);
				}
				else if (xImageSrc.IsOfType<Clu::TPixel_Lum_UInt16>()
					&& xImageMask.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					Kernel::MaskToBlack<Clu::TPixel_Lum_UInt16, Clu::TPixel_Lum_UInt8>
						CLU_KERNEL_CONFIG()
						(xImageTrg, xImageSrc, xImageMask);
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}

			}





		} // ImgProc
	} // Cuda
} // Clu


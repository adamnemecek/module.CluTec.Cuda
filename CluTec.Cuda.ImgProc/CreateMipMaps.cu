////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      CreateMipMaps.cu
//
// summary:   create mip maps class
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

#include "CreateMipMaps.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Cuda.Base/Conversion.h"


namespace Clu
{
	namespace Cuda
	{
		namespace CreateMipMaps
		{
			namespace Kernel
			{
				struct Const_H
				{
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 4;
					static const int ThreadCountY = 8;
					static const int SrcBlockSizeX = 2 * ThreadCountX;
					static const int SrcBlockSizeY = ThreadCountY;
					static const int TrgBlockSizeX = ThreadCountX;
					static const int TrgBlockSizeY = ThreadCountY;
				};

				struct Const_V
				{
					static const int WarpsPerBlockX = 1;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 4;
					static const int SrcBlockSizeX = ThreadCountX;
					static const int SrcBlockSizeY = 2 * ThreadCountY;
					static const int TrgBlockSizeX = ThreadCountX;
					static const int TrgBlockSizeY = ThreadCountY;
				};


				template<typename TPixel>
				__global__ void SmoothHorizontalLum(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixel>::TElement;

					const int iSrcX = int(blockIdx.x * Const_H::SrcBlockSizeX) 
									+ int((threadIdx.x % Const_H::ThreadCountX) * 2);

					const int iSrcY = int(blockIdx.y * Const_H::SrcBlockSizeY) 
									+ int((threadIdx.x / Const_H::ThreadCountX));
					
					const int iTrgX = int(blockIdx.x * Const_H::TrgBlockSizeX)
									+ int((threadIdx.x % Const_H::ThreadCountX));

					const int iTrgY = int(blockIdx.y * Const_H::TrgBlockSizeY)
									+ int((threadIdx.x / Const_H::ThreadCountX));
					
					if (!xImageSrc.IsRectInside(iSrcX, iSrcY, 3, 1))
					{
						return;
					}

					unsigned uValue = (unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX, iSrcY).r();
					uValue += ((unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX + 1, iSrcY).r()) << 1;
					uValue += (unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX + 2, iSrcY).r();
					uValue >>= 2;

					xImageTrg.Write2D(Clu::Cuda::Make<TElement>(uValue), iTrgX, iTrgY);
				}


				template<typename TPixel>
				__global__ void SmoothVerticalLum(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixel>::TElement;

					const int iSrcX = int(blockIdx.x * Const_V::SrcBlockSizeX)
						+ int((threadIdx.x % Const_V::ThreadCountX));

					const int iSrcY = int(blockIdx.y * Const_V::SrcBlockSizeY)
						+ int((threadIdx.x / Const_V::ThreadCountX) * 2);

					const int iTrgX = int(blockIdx.x * Const_V::TrgBlockSizeX)
						+ int((threadIdx.x % Const_V::ThreadCountX));

					const int iTrgY = int(blockIdx.y * Const_V::TrgBlockSizeY)
						+ int((threadIdx.x / Const_V::ThreadCountX));

					if (!xImageSrc.IsRectInside(iSrcX, iSrcY, 1, 3))
					{
						return;
					}

					unsigned uValue = (unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX, iSrcY).r();
					uValue += ((unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX, iSrcY + 1).r()) << 1;
					uValue += (unsigned)xImageSrc.ReadPixel2D<TPixel>(iSrcX, iSrcY + 2).r();
					uValue >>= 2;

					xImageTrg.Write2D(Clu::Cuda::Make<TElement>(uValue), iTrgX, iTrgY);
				}




			} // namespace Kernel



			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				Clu::SImageFormat xF(xFormat);

				EvalThreadConfigBlockSize(xDevice, xF
					, Kernel::Const_H::SrcBlockSizeX, Kernel::Const_H::SrcBlockSizeY, 0, 1, 0, 0
					, Kernel::Const_H::WarpsPerBlockX, Kernel::Const_H::WarpsPerBlockY
					, NumberOfRegisters, false
					, KernelSmoothHorizontal);

				xF.iWidth = Clu::Cuda::_CDeviceSurfMipMap::NextLevelWidth(xF.iWidth);

				EvalThreadConfigBlockSize(xDevice, xF
					, Kernel::Const_V::SrcBlockSizeX, Kernel::Const_V::SrcBlockSizeY, 0, 0, 0, 1
					, Kernel::Const_V::WarpsPerBlockX, Kernel::Const_V::WarpsPerBlockY
					, NumberOfRegisters, false
					, KernelSmoothVertical);
			}




			double CDriver::Run(Clu::Cuda::CDeviceSurfMipMap& xMipMap, Clu::Cuda::CDeviceSurface& xTemp, int iLevelCount)
			{
				try
				{
					Clu::CIString sHName = "Create Mip-Maps -> Smooth Horizontal";
					Clu::CIString sVName = "Create Mip-Maps -> Smooth Vertical";
					double dTotalProcessTime = 0.0;

					if (iLevelCount > xMipMap.MaxMipMapLevelCount())
					{
						throw CLU_EXCEPTION("Mip-map level count larger than the maximal possible level");
					}

					if (iLevelCount < 0)
					{
						iLevelCount = xMipMap.MaxMipMapLevelCount();
					}
					else if (iLevelCount == 1 || iLevelCount == 0)
					{
						return 0.0;
					}

					xMipMap.SetActiveMipMapLevelCount(iLevelCount);

					if (xMipMap.IsOfType<TPixel_Lum_UInt8>())
					{
						for (int iLevel = 0; iLevel < iLevelCount - 1; ++iLevel)
						{
							Clu::_SImageFormat xF = xMipMap.Format(iLevel);
							xF.iWidth = Clu::Cuda::_CDeviceSurfMipMap::NextLevelWidth(xF.iWidth);

							xTemp.Create(xF);

							SafeBegin(sHName);
							Kernel::SmoothHorizontalLum<TPixel_Lum_UInt8>
								CLU_KERNEL_CONFIG_(KernelSmoothHorizontal)
								(xTemp, xMipMap.GetMipMap(iLevel));
							SafeEnd(sHName);
							LogLastProcessTime(sHName);
							dTotalProcessTime += LastProcessTime();

							SafeBegin(sVName);
							Kernel::SmoothVerticalLum<TPixel_Lum_UInt8>
								CLU_KERNEL_CONFIG_(KernelSmoothVertical)
								(xMipMap.GetMipMap(iLevel + 1), xTemp);
							SafeEnd(sVName);
							LogLastProcessTime(sVName);
							dTotalProcessTime += LastProcessTime();
						}
					}

					CLU_LOG(CLU_S "Create Mip-Maps total process time: " << dTotalProcessTime << "\n");
					return dTotalProcessTime;
				}
				CLU_CATCH_RETHROW_ALL("Error creating mip-maps")
			}



		} // ImgProc
	} // Cuda
} // Clu


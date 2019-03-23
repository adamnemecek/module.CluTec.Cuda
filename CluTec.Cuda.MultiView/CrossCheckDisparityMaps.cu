////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      CrossCheckDisparityMaps.cu
//
// summary:   cross check disparity maps class
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

#include "CrossCheckDisparityMaps.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace CrossCheckDisparityMaps
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
					static const int ThreadsPerBlockX = WarpsPerBlockX * 32;
				};

				__constant__ _SParameter c_xPars;

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Transform image. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageTrg">	The image trg. </param>
				/// <param name="xImageSrc">	The image source. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixel>
				__global__ void Transform(Clu::Cuda::_CDeviceSurface surfDispResult
					, Clu::Cuda::_CDeviceImage deviRange
					, const Clu::Cuda::_CDeviceSurface surfDispLR
					, const Clu::Cuda::_CDeviceSurface surfDispRL)
				{
					using TData = typename TPixel::TData;

					int iLeftX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int iLeftY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!surfDispLR.IsInside(iLeftX, iLeftY, 1, 1))
					{
						return;
					}

					TPixel pixDispL = surfDispLR.ReadPixel2D<TPixel>(iLeftX, iLeftY);
					const TData xDispL = pixDispL.r();

					if (xDispL < TDisp(EDisparityId::First) || xDispL > TDisp(EDisparityId::Last))
					{
						surfDispResult.WritePixel2D<TPixel>(pixDispL, iLeftX, iLeftY);
						return;
					}

					int iRightX = c_xPars.xDispConfig.MapPixelPos(iLeftX, xDispL - TDisp(EDisparityId::First), c_xPars.iIsLeftToRight);
					int iRightY = iLeftY;

					if (!surfDispRL.IsInside(iRightX, iRightY))
					{
						pixDispL.r() = TDisp(EDisparityId::Inconsistent);
						surfDispResult.WritePixel2D<TPixel>(pixDispL, iLeftX, iLeftY);
						return;
					}

					TPixel pixDispR = surfDispRL.ReadPixel2D<TPixel>(iRightX, iRightY);
					if (abs(int(pixDispR.r()) - int(xDispL)) <= c_xPars.iDispDeltaThresh)
					{
						surfDispResult.WritePixel2D<TPixel>(pixDispL, iLeftX, iLeftY);

						atomicMin(&deviRange.At<TPixelRange>(threadIdx.x, 0).r(), TRange(pixDispL.r()));
						atomicMax(&deviRange.At<TPixelRange>(threadIdx.x, 1).r(), TRange(pixDispL.r()));
					}
					else
					{
						pixDispL.r() = TDisp(EDisparityId::Inconsistent);
						surfDispResult.WritePixel2D<TPixel>(pixDispL, iLeftX, iLeftY);
					}
				}

			}



			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
				, const SParameter& xPars)
			{
				m_xPars = xPars;

				// Reserve memory for min-max data
				Clu::SImageFormat xF(Kernel::Const::ThreadsPerBlockX, 2, Clu::SImageType(TPixelRange::PixelTypeId, TPixelRange::DataTypeId));
				m_imgRange.Create(xF);
				m_deviRange.Create(xF);

				EvalThreadConfigBlockSize(xDevice, xFormat
					, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
					, 0, 0, 0, 0 // Offsets
					, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
					, NumberOfRegisters
					, false // Use also partial blocks
					);
			}


			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xDispResult
				, const Clu::Cuda::_CDeviceSurface& xDispLR
				, const Clu::Cuda::_CDeviceSurface& xDispRL)
			{
				if (!xDispLR.IsEqualFormat(xDispRL.Format()))
				{
					throw CLU_EXCEPTION("Input disparity images have different formats");
				}

				if (!xDispLR.IsEqualFormat(xDispResult.Format()))
				{
					throw CLU_EXCEPTION("Input and output disparity images have different formats");
				}
				TPixelRange *pData = (TPixelRange*)m_imgRange.DataPointer();
				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					pData->r() = TRange(EDisparityId::Last);
				}

				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					pData->r() = TRange(EDisparityId::First);
				}

				m_deviRange.CopyFrom(m_imgRange);

				
				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);


				if (xDispLR.IsOfType<TPixelDisp>()
					&& xDispRL.IsOfType<TPixelDisp>()
					&& xDispResult.IsOfType<TPixelDisp>())
				{
					Kernel::Transform<TPixelDisp>
						CLU_KERNEL_CONFIG()
						(xDispResult, m_deviRange, xDispLR, xDispRL);
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}

				m_deviRange.CopyInto(m_imgRange);
				m_xPars.ResetRange();

				pData = (TPixelRange *)m_imgRange.DataPointer();
				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					m_xPars.uDispMin = Clu::Min(m_xPars.uDispMin, TDisp(pData->r()));
				}

				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					m_xPars.uDispMax = Clu::Max(m_xPars.uDispMax, TDisp(pData->r()));
				}

			}





		} // ImgProc
	} // Cuda
} // Clu


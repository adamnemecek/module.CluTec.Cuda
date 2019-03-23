////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Statistics.Mean.cu
//
// summary:   statistics. mean class
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

#include "Statistics.Mean.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

#include "CluTec.Cuda.Base/Kernel.ArrayCache.h"
#include "CluTec.Cuda.Base/Kernel.Debug.h"
#include "Kernel.Algo.ArraySum.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Statistics
		{
			namespace Mean
			{
				namespace Kernel
				{
					using namespace Clu::Cuda::Kernel;

					template<int t_iPatchWidth, int t_iPatchHeight
						, int t_iPatchCountY_Pow2
						, int t_iWarpsPerBlockX, int t_iWarpsPerBlockY>
					struct Constants
					{
						// Warps per block
						static const int WarpsPerBlockX = t_iWarpsPerBlockX;
						static const int WarpsPerBlockY = t_iWarpsPerBlockY;

						// Thread per warp
						static const int ThreadsPerWarp = 32;

						static const int ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;

						// the width of a base patch has to be a full number of words
						static const int BasePatchSizeX = t_iPatchWidth;
						static const int BasePatchSizeY = t_iPatchHeight;
						static const int BasePatchElementCount = BasePatchSizeX * BasePatchSizeY;

						using AlgoSum = Clu::Cuda::Kernel::AlgoArraySum<ThreadsPerBlockX, t_iPatchCountY_Pow2,
							BasePatchSizeX, BasePatchSizeY>;

						static const int BasePatchCountX = AlgoSum::PatchCountX;
						static const int BasePatchCountY = AlgoSum::PatchCountY;

						static const int TestBlockSizeX = AlgoSum::DataArraySizeX + 1;
						static const int TestBlockSizeY = AlgoSum::DataArraySizeY;

						static const int SumCacheSizeX = AlgoSum::SumCacheSizeX;
						static const int SumCacheSizeY = AlgoSum::SumCacheSizeY;

						static const int ResultCountPerThread = AlgoSum::ColGroupSizeX;


#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

						__device__ static void PrintValues()
						{
							printf("Block Idx: %d, %d\n", blockIdx.x, blockIdx.y);
							PRINT(ThreadsPerBlockX);
							PRINT(BasePatchSizeX);
							PRINT(BasePatchSizeY);
							PRINT(BasePatchCountX);
							PRINT(BasePatchCountY);
							PRINT(TestBlockSizeX);
							PRINT(TestBlockSizeY);
							PRINT(ResultCountPerThread);
							//PRINT();
							printf("\n");

							__syncthreads();
						}
#undef PRINT
					};

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	The algorithm parameters. </summary>
					///
					/// <value>	. </value>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					__constant__ _SParameter c_xPars;

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Transform image. </summary>
					///
					/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TPixel, EConfig t_eConfig>
					__global__ void Transform(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
					{
						using Config = SConfig<t_eConfig>;
						using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::PatchCountY_Pow2, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

						using AlgoSum = typename Const::AlgoSum;
						using TData = typename TPixel::TData;
						using TSum = short;// unsigned short;

						// ////////////////////////////////////////////////////////////////////////////////////////////////
						// ////////////////////////////////////////////////////////////////////////////////////////////////
						// ////////////////////////////////////////////////////////////////////////////////////////////////

						// Get position of thread in left image
						const int iBlockX = blockIdx.x * Const::BasePatchCountX;
						const int iBlockY = blockIdx.y * Const::BasePatchCountY;

						if (!xImageSrc.Format().IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
						{
							return;
						}

						// ////////////////////////////////////////////////////////////////////////////////////////////////
						// ////////////////////////////////////////////////////////////////////////////////////////////////
						// ////////////////////////////////////////////////////////////////////////////////////////////////
						Debug::Run([]()
						{
							if (Debug::IsThreadAndBlock(0, 0, 0, 0))
							{
								Const::PrintValues();
							}
						});

						// ////////////////////////////////////////////////////////////////////////////////////////////////
						// ////////////////////////////////////////////////////////////////////////////////////////////////
						__shared__ Clu::Cuda::Kernel::CArrayCache<TSum
							, Const::SumCacheSizeX, Const::SumCacheSizeY
							, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>
							xCacheSumY;


						TSum puMean[Const::ResultCountPerThread];

						Const::AlgoSum::SumCache(xCacheSumY,
							[&xImageSrc, &iBlockX, &iBlockY](int iIdxX, int iIdxY)
						{
							return (TSum) xImageSrc.Read2D<TData>(iBlockX + iIdxX, iBlockY + iIdxY);
						},
							[&puMean](TSum& iValue, int iResultIdx, int iIdxX, int iIdxY)
						{

							puMean[iResultIdx] = iValue / Const::BasePatchElementCount;
						});


						for (int iResultIdx = 0; iResultIdx < Const::ResultCountPerThread; ++iResultIdx)
						{
							TData uValue = (TData)min(puMean[iResultIdx], (TSum)Clu::NumericLimits<TData>::Max());

							xImageTrg.Write2D(uValue
								, AlgoSum::RowSumBaseIdxX() + iBlockX + Const::BasePatchSizeX / 2 + iResultIdx
								, AlgoSum::RowSumBaseIdxY() + iBlockY + Const::BasePatchSizeY / 2);
						}
					}

				}

				template<EConfig t_eConfig>
				void CDriver::_DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::PatchCountY_Pow2, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					unsigned nOffsetLeft	= 0; //Config::PatchSizeX / 2;
					unsigned nOffsetRight	= 0; //Config::PatchSizeX / 2;
					unsigned nOffsetTop		= 0; //Config::PatchSizeY / 2;
					unsigned nOffsetBottom	= 0; //Config::PatchSizeY / 2;

					EvalThreadConfigBlockSize(xDevice, xFormat
						, Const::BasePatchCountX, Const::BasePatchCountY
						, nOffsetLeft, nOffsetRight, nOffsetTop, nOffsetBottom
						, Config::WarpsPerBlockX, Config::WarpsPerBlockY
						, Config::NumberOfRegisters
						, true // do not process partial blocks
						);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Configures. </summary>
				///
				/// <param name="xConfig">	[in,out] The configuration. </param>
				/// <param name="xDevice">	The device. </param>
				/// <param name="xFormat">	Describes the format to use. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

#define _CLU_DO_CONFIG(theId) \
			case theId: \
				_DoConfigure<theId>(xDevice, xFormat); \
				break

				void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
					const SParameter& xPars)
				{
					m_xPars = xPars;

					switch (m_xPars.eConfig)
					{
						_CLU_DO_CONFIG(EConfig::Patch_16x16);
						_CLU_DO_CONFIG(EConfig::Patch_11x11);
						_CLU_DO_CONFIG(EConfig::Patch_9x9);
						_CLU_DO_CONFIG(EConfig::Patch_7x7);
						_CLU_DO_CONFIG(EConfig::Patch_5x5);
						_CLU_DO_CONFIG(EConfig::Patch_3x3);

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}

				}
#undef _CLU_DO_CONFIG

#define _CLU_DO_PROCESS(theId) \
			case theId: \
				Kernel::Transform<TPixel, theId> \
					CLU_KERNEL_CONFIG() \
					(xImageOut, xImageIn); \
				break

				template<typename TPixel>
				void CDriver::_DoProcess(Clu::Cuda::_CDeviceSurface& xImageOut
					, const Clu::Cuda::_CDeviceSurface& xImageIn)
				{
					switch (m_xPars.eConfig)
					{
						_CLU_DO_PROCESS(EConfig::Patch_16x16);
						_CLU_DO_PROCESS(EConfig::Patch_11x11);
						_CLU_DO_PROCESS(EConfig::Patch_9x9);
						_CLU_DO_PROCESS(EConfig::Patch_7x7);
						_CLU_DO_PROCESS(EConfig::Patch_5x5);
						_CLU_DO_PROCESS(EConfig::Patch_3x3);

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}

				}
#undef _CLU_DO_PROCESS

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Process this object. </summary>
				///
				/// <param name="dimBlocksInGrid">   	The dim blocks in grid. </param>
				/// <param name="dimThreadsPerBlock">	The dim threads per block. </param>
				/// <param name="xImageDisp">		 	[in,out] The image disp. </param>
				/// <param name="xImageL">			 	The image l. </param>
				/// <param name="xImageR">			 	The image r. </param>
				/// <param name="iOffset">			 	Zero-based index of the offset. </param>
				/// <param name="iDispRange">		 	Zero-based index of the disp range. </param>
				/// <param name="fSadThresh">		 	The sad thresh. </param>
				/// <param name="iMinDeltaThresh">   	Zero-based index of the minimum delta thresh. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageOut
					, const Clu::Cuda::_CDeviceSurface& xImageIn)
				{
					Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

					if (!xImageOut.IsEqualType(xImageIn.Format()))
					{
						throw CLU_EXCEPTION("Given output image has different type than input image");
					}

					if (xImageIn.IsOfType<Clu::TPixel_Lum_UInt8>()
						&& xImageOut.IsOfType<Clu::TPixel_Lum_UInt8>())
					{
						_DoProcess<Clu::TPixel_Lum_UInt8>(xImageOut, xImageIn);
					}
					else
					{
						throw CLU_EXCEPTION("Pixel types of given images not supported");
					}
				}


				// /////////////////////////////////////////////////////////////////////////
				// Snippet: calculate variance
				// unsigned char pucIsBelowVarThresh[Const::BasePatchCountSplitX];
				//
				// {
				//
				//	// Evaluate mean value of base patches.
				//	Const::AlgoSum::SumCache(xCacheSumY,
				//		[&xCacheBasePatch](int iIdxX, int iIdxY)
				//	{
				//		auto pixVal = xCacheBasePatch.At(iIdxX, iIdxY);
				//		return TSum(pixVal.x);
				//	},
				//		[&pxMinDisp](TSum& iValue, int iRelX, int iIdxX, int iIdxY)
				//	{
				//		pxMinDisp[iRelX].x = iValue / Const::BasePatchElementCount;
				//	}
				//	);
				//
				//	// Evaluate mean deviation from mean value
				//	Const::AlgoSum::SumCache(xCacheSumY,
				//		[&xCacheBasePatch](int iIdxX, int iIdxY)
				//	{
				//		auto pixVal = xCacheBasePatch.At(iIdxX, iIdxY);
				//		return TSum(pixVal.x >> 4) * TSum(pixVal.x >> 4);
				//	},
				//		[&pxMinDisp](TSum& iValue, int iRelX, int iIdxX, int iIdxY)
				//	{
				//		pxMinDisp[iRelX].y = iValue / Const::BasePatchElementCount;
				//	}
				//	);
				//
				//	int iIsBelowVarThresh = 1;
				//	for (int iRelX = 0; iRelX < Const::BasePatchCountSplitX; ++iRelX)
				//	{
				//		float fMean = float(pxMinDisp[iRelX].x);
				//		float fMean2 = float(pxMinDisp[iRelX].y);
				//		float fVar = fMean2 - fMean * fMean;
				//
				//		//if (IsBlock(40, 40))
				//		//{
				//		//	printf("%g\n", fVar);
				//		//}
				//
				//		int iIsBelow = int(fVar < 5.0f);
				//		pucIsBelowVarThresh[iRelX] = (unsigned char)iIsBelow;
				//		iIsBelowVarThresh *= iIsBelow;
				//
				//		//static const float fVarMax = 5000.0f;
				//		//fVar /= fVarMax;
				//
				//		////TDispComp ucValue = Clu::NormFloatTo<TDispComp>(fVar);
				//		//TDisp pixDisp = Clu::Cuda::NormFloatToColor<TPixelDisp>(fVar);
				//
				//		//surfDisp.WritePixel2D<TPixelDisp>(pixDisp //make_uchar4(ucValue, ucValue, ucValue, 255)
				//		//	, AlgoSum::BaseIdxX() + iDispX + iRelX
				//		//	, AlgoSum::BaseIdxY() + iDispY);
				//	}
				//
				//	if (__syncthreads_and(iIsBelowVarThresh) > 0)
				//	{
				//		return;
				//	}
				//
				// }




			} // Mean
		} // Statistics
	} // Cuda
} // Clu


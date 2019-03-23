////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherSAD1.cu
//
// summary:   block matcher sad 1 class
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
#include "CluTec.Types1/Pixel.h"

#include "BlockMatcherSAD1.h"

#include "CluTec.Math/Conversion.h"
#include "CluTec.Cuda.Base/PixelTypeInfo.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/Conversion.h"

#define DEBUG_BLOCK_X 40
#define DEBUG_BLOCK_Y 42
//#define CLU_DEBUG_CACHE
#define CLU_DEBUG_KERNEL

#include "CluTec.Cuda.Base/Kernel.ArrayCache.h"
#include "CluTec.Cuda.Base/Kernel.Debug.h"
#include "CluTec.Cuda.ImgProc/Kernel.Algo.ArraySum.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"

namespace Clu
{
	namespace Cuda
	{
		namespace BlockMatcherSAD1
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

					static const int BasePatchCountSplitX = AlgoSum::PatchCountSplitX;


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
						PRINT(BasePatchCountSplitX);
						//PRINT();
						printf("\n");

						__syncthreads();
					}
#undef PRINT
				};


				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Eval sad. </summary>
				///
				/// <typeparam name="Const">		  	Type of the constant. </typeparam>
				/// <typeparam name="TPixelSurf">	  	Type of the pixel surf. </typeparam>
				/// <typeparam name="TMinRange">	  	Type of the minimum range. </typeparam>
				/// <typeparam name="TDispComp">	  	Type of the disp component. </typeparam>
				/// <typeparam name="TCacheBasePatch">	Type of the cache base patch. </typeparam>
				/// <typeparam name="TCacheSum">	  	Type of the cache sum. </typeparam>
				/// <param name="pxMinRange">	  	[in,out] If non-null, the px minimum range. </param>
				/// <param name="pxDisp">		  	[in,out] If non-null, the px disp. </param>
				/// <param name="xCacheBasePatch">	[in,out] The cache base patch. </param>
				/// <param name="xCacheSumY">	  	[in,out] The cache sum y coordinate. </param>
				/// <param name="surfImgR">		  	[in,out] The surf image r. </param>
				/// <param name="iBlockX">		  	Zero-based index of the block x coordinate. </param>
				/// <param name="iBlockY">		  	Zero-based index of the block y coordinate. </param>
				/// <param name="iDispIdx">		  	Zero-based index of the disp index. </param>
				/// <param name="fSadThresh">	  	The sad thresh. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename Const, typename TPixelSurf, typename TMinRange, typename TDispComp, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalSAD(TMinRange* pxMinRange
					, TDispComp* pxDisp
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY
					, Clu::Cuda::_CDeviceSurface& surfImgR
					, int iBlockX, int iBlockY, int iDispIdx, const int iDispIdxInc, float fSadThresh)
				{
					using TSum = typename TCacheSum::TElement;

					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch, &surfImgR, iBlockX, iBlockY](int iIdxX, int iIdxY)
					{
						auto pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						auto pixVal2 = surfImgR.ReadPixel2D<TPixelSurf>(iBlockX + iIdxX, iBlockY + iIdxY);
						return (TSum)abs(int(pixVal1.x) - int(pixVal2.r()));
					},
						[&pxMinRange, &pxDisp, &iDispIdx, &iDispIdxInc, &fSadThresh](TSum& iValue, int iRelX, int iIdxX, int iIdxY)
					{
						if (pxDisp[iRelX] == TDisp(EDisparityId::CannotEvaluate))
						{
							return;
						}

						auto& xMinRange = pxMinRange[iRelX];

						float fMin = float(iValue) / float(Const::BasePatchSizeX * Const::BasePatchSizeY);

						// xMinRange.x: Minimum value
						// xMinRange.w: Value at previous disparity step
						// xMinRange.y: Value at disparity step just before minimum
						// xMinRange.z: Value at disparity step just after minimum
						if (iDispIdx == 0)
						{
							xMinRange.w = iValue;
							xMinRange.y = iValue;
						}

						int iIsBelowMin = int(iValue < xMinRange.x && fMin < fSadThresh);

						if (iIsBelowMin)
						{
							// We have a new minimum, so set the value before minimum to previous value.
							xMinRange.y = xMinRange.w;

							// Set the minimum
							xMinRange.x = iValue;

							// Set the value after the minimum also to the minimum.
							// Can only be set properly in the next step.
							xMinRange.z = iValue;

							// Set disparity at minimum
							pxDisp[iRelX] = iDispIdx;
						}
						else
						{
							// The current value is not a new minimum.
							// If the previous value is the minimum then set the value after the minimum to this value.
							if (pxDisp[iRelX] + iDispIdxInc == iDispIdx)
							{
								xMinRange.z = iValue;
							}
						}

						// Set current value to previous value
						xMinRange.w = iValue;

						//xMinRange.x = max(iValue * iIsBelowMin, xMinRange.x * (1 - iIsBelowMin));
						//pxDisp[iRelX] = max(pxDisp[iRelX] * (1 - iIsBelowMin), iDispIdx * iIsBelowMin);
					}
					);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Eval graduated. </summary>
				///
				/// <typeparam name="Const">		  	Type of the constant. </typeparam>
				/// <typeparam name="TDispComp">	  	Type of the disp component. </typeparam>
				/// <typeparam name="TCacheBasePatch">	Type of the cache base patch. </typeparam>
				/// <typeparam name="TCacheSum">	  	Type of the cache sum. </typeparam>
				/// <param name="pxDisp">		  	[in,out] If non-null, the px disp. </param>
				/// <param name="xCacheBasePatch">	[in,out] The cache base patch. </param>
				/// <param name="xCacheSumY">	  	[in,out] The cache sum y coordinate. </param>
				/// <param name="fSadThresh">	  	The sad thresh. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename Const, typename TDispComp, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalGrad(TDispComp* pxDisp
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY
					, float fGradThresh)
				{
					using TSum = typename TCacheSum::TElement;

					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch](int iIdxX, int iIdxY)
					{
						auto pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						auto pixVal2 = xCacheBasePatch.At(iIdxX + 1, iIdxY);
						return (TSum)abs(int(pixVal1.x) - int(pixVal2.x));
					},
						[&pxDisp, &fGradThresh](TSum& iValue, int iRelX, int iIdxX, int iIdxY)
					{
						float fValue = float(iValue) / float(Const::BasePatchElementCount);

						if (fValue < fGradThresh)
						{
							pxDisp[iRelX] = TDisp(EDisparityId::CannotEvaluate);
						}
					});
				}




				__constant__ _SParameter c_xPars;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDisp;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgL;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgR;

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Matchers. </summary>
				///
				/// <typeparam name="TPixelDisp">		  	Type of the pixel disp. </typeparam>
				/// <typeparam name="TPixelSrc">		  	Type of the pixel source. </typeparam>
				/// <typeparam name="t_iPatchWidth">	  	Type of the patch width. </typeparam>
				/// <typeparam name="t_iPatchHeight">	  	Type of the patch height. </typeparam>
				/// <typeparam name="t_iPatchCountY_Pow2">	Type of the i patch count y coordinate pow 2. </typeparam>
				/// <typeparam name="t_iWarpsPerBlockX">  	Type of the warps per block x coordinate. </typeparam>
				/// <typeparam name="t_iWarpsPerBlockY">  	Type of the warps per block y coordinate. </typeparam>
				/// <param name="surfDisp">		  	The surf disp. </param>
				/// <param name="surfImgL">		  	The surf image l. </param>
				/// <param name="surfImgR">		  	The surf image r. </param>
				/// <param name="iOffset_px">	  	Zero-based index of the offset px. </param>
				/// <param name="iDispRange">	  	Zero-based index of the disp range. </param>
				/// <param name="fSadThresh">	  	The sad thresh. </param>
				/// <param name="iMinDeltaThresh">	Zero-based index of the minimum delta thresh. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig>
					__global__ void Matcher()
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::PatchCountY_Pow2, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoSum = typename Const::AlgoSum;

					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;
					using TComponent = typename TPixelSrc::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TSum = short;// unsigned short;
					using TMinRange = ushort4;
					using TDispRange = unsigned short;


					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////

					// Get position of thread in left image
					const int iBlockX = c_xPars._iBlockOffsetX + blockIdx.x * Const::BasePatchCountX;
					const int iBlockY = c_xPars._iBlockOffsetY + blockIdx.y * Const::BasePatchCountY;

					if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
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
					// The Cache. Always a set of uints to speed up reading.
					__shared__ Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>
					xCacheBasePatch;

					__shared__ Clu::Cuda::Kernel::CArrayCache<TSum
						, Const::SumCacheSizeX, Const::SumCacheSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>
					xCacheSumY;

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Read an image area into the cache
					xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
					__syncthreads();

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Initialize the minimum SAD and Disparity arrays
					TMinRange pxMinRange[Const::BasePatchCountSplitX];
					TDispRange pxDisp[Const::BasePatchCountSplitX];

#					pragma unroll
					for (int i = 0; i < Const::BasePatchCountSplitX; ++i)
					{
						pxMinRange[i] = make_ushort4(0xFFFF, 0, 0, 0);
						pxDisp[i] = TDisp(EDisparityId::Unknown);
					}

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Evaluate block gradient in left image to identify patches that are distinguishable from their neighbors
					EvalGrad<Const>(pxDisp, xCacheBasePatch, xCacheSumY, c_xPars.fGradThresh);

					int iDoEvalDisp = 0;
					for (int i = 0; i < Const::BasePatchCountSplitX; ++i)
					{
						if (pxDisp[i] != TDisp(EDisparityId::CannotEvaluate))
						{
							iDoEvalDisp = 1;
							break;
						}
					}

					if (!__syncthreads_or(iDoEvalDisp))
					{
						TDispTuple pixValue;
						pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::CannotEvaluate), 0, 0, 0);

						for (int i = 0; i < Const::BasePatchCountSplitX; ++i)
						{
							c_surfDisp.Write2D(pixValue
								, AlgoSum::RowSumBaseIdxX() + iBlockX + Const::BasePatchSizeX / 2 + i
								, AlgoSum::RowSumBaseIdxY() + iBlockY + Const::BasePatchSizeY / 2);
						}

						return;
					}


					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Start disparity evaluation
					static const int c_iInitialDispInc = 4;

					for (int iDispIdx = 0; iDispIdx < c_xPars.iDispRange; iDispIdx += c_iInitialDispInc)
					{
						const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange + iDispIdx;

						if (c_surfImgR.Format().IsRectInside(iTestX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
						{
							EvalSAD<Const, TPixelSrc>(pxMinRange, pxDisp, xCacheBasePatch, xCacheSumY, c_surfImgR
								, iTestX, iBlockY
								, iDispIdx, c_iInitialDispInc, c_xPars.fSadThresh);
						}
					}

					__syncthreads();

					for (int iDispIdx = 0; iDispIdx < c_xPars.iDispRange; ++iDispIdx)
					{
						const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange + iDispIdx;

						if (!c_surfImgR.Format().IsRectInside(iTestX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
						{
							continue;
						}

						int iIsInside = 0;
						for (int i = 0; i < Const::BasePatchCountSplitX; ++i)
						{
							int iValue = (iDispIdx + c_iInitialDispInc >= int(pxDisp[i])
								&& iDispIdx <= int(pxDisp[i]) + c_iInitialDispInc);

							//if (iIsDebugBlock && iValue)
							//{
							//	printf("%d: %d => %d\n", iDispIdx, pxMinDisp[i].y, iValue);
							//}

							iIsInside += iValue;
						}

						if (__syncthreads_or(iIsInside) == 0)
						{
							continue;
						}

						EvalSAD<Const, TPixelSrc>(pxMinRange, pxDisp, xCacheBasePatch, xCacheSumY, c_surfImgR
							, iTestX, iBlockY, iDispIdx, 1, c_xPars.fSadThresh);
					}

					__syncthreads();


					for (int iRelX = 0; iRelX < Const::BasePatchCountSplitX; ++iRelX)
					{
						TMinRange& xMinRange = pxMinRange[iRelX];


						float fMin = float(xMinRange.x) / float(Const::BasePatchElementCount);

						int iIsBelowMin = int(fMin < c_xPars.fSadThresh);
						int iDiff1 = abs(int(xMinRange.y) - int(xMinRange.x));
						int iDiff2 = abs(int(xMinRange.z) - int(xMinRange.x));

						//if (IsBlock(40, 40))
						//{
						//	printf("%d, %d\n", iDiff1, iDiff2);
						//}

						TDispTuple pixValue;


						if (pxDisp[iRelX] > TDisp(EDisparityId::Last))
						{
							pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDisp[iRelX], 0, 0, 0);
						}
						else if (iIsBelowMin)
						{
							int iIsDispSaturated = (pxDisp[iRelX] <= TDisp(EDisparityId::First) + 1 || pxDisp[iRelX] >= TDisp(EDisparityId::First) + c_xPars.iDispRange - 1);

							if (iDiff1 > c_xPars.iMinDeltaThresh && iDiff2 > c_xPars.iMinDeltaThresh && iIsDispSaturated == 0)
							{
								pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::First) + pxDisp[iRelX], xMinRange.x / Const::BasePatchElementCount, iDiff1, iDiff2);
							}
							else if (iIsDispSaturated)
							{
								pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::Saturated), xMinRange.x / Const::BasePatchElementCount, iDiff1, iDiff2);
							}
							else
							{
								pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotSpecific), xMinRange.x / Const::BasePatchElementCount, iDiff1, iDiff2);
							}
						}
						else
						{
							pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotFound), xMinRange.x / Const::BasePatchElementCount, iDiff1, iDiff2);
						}

						c_surfDisp.Write2D(pixValue
							, AlgoSum::RowSumBaseIdxX() + iBlockX + Const::BasePatchSizeX / 2 + iRelX
							, AlgoSum::RowSumBaseIdxY() + iBlockY + Const::BasePatchSizeY / 2);
					}

				}
			}


			template<EConfig t_eConfig>
			void CDriver::_DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				using Config = SConfig<t_eConfig>;
				using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::PatchCountY_Pow2, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

				unsigned nByteOffset = (m_xPars.iOffset) * (unsigned)xFormat.BytesPerPixel();
				unsigned nOffsetLeft = ((nByteOffset / 128 + (nByteOffset % 128 > 0 ? 1 : 0)) * 128) / (unsigned)xFormat.BytesPerPixel();

				unsigned nOffsetRight = Const::TestBlockSizeX - Const::BasePatchCountX;

				unsigned nOffsetTop = 0;
				unsigned nOffsetBottom = Const::TestBlockSizeY - Const::BasePatchCountY;

				m_xPars._iBlockOffsetX = (int)nOffsetLeft;
				m_xPars._iBlockOffsetY = 0;
				m_xPars._iRightToLeftOffsetX = (int)nOffsetLeft;

				EvalThreadConfigBlockSize(xDevice, xFormat
					, Const::BasePatchCountX, Const::BasePatchCountY
					, nOffsetLeft, nOffsetRight, nOffsetTop, nOffsetBottom
					, Config::WarpsPerBlockX, Config::WarpsPerBlockY
					, Config::NumberOfRegisters);
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Configures. </summary>
			///
			/// <param name="xConfig">	[in,out] The configuration. </param>
			/// <param name="xDevice">	The device. </param>
			/// <param name="xFormat">	Describes the format to use. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
				const SParameter& xPars)
			{
				m_xPars = xPars;

				switch (m_xPars.eConfig)
				{
				case EConfig::Patch_16x16:
					_DoConfigure<EConfig::Patch_16x16>(xDevice, xFormat);
					break;

				default:
					throw CLU_EXCEPTION("Invalid algorithm configuration.");
				}

			}


			template<typename TPixelDisp, typename TPixelSrc>
			void CDriver::_DoProcess()
			{
				switch (m_xPars.eConfig)
				{
				case EConfig::Patch_16x16:
					Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_16x16>
						CLU_KERNEL_CONFIG()();
					break;

				default:
					throw CLU_EXCEPTION("Invalid algorithm configuration.");
				}

			}

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

			void CDriver::Process(Clu::Cuda::CDeviceSurface& xImageDisp
				, const Clu::Cuda::CDeviceSurface& xImageL
				, const Clu::Cuda::CDeviceSurface& xImageR)
			{
				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDisp, &xImageDisp, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfImgL, &xImageL, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfImgR, &xImageR, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

				if (!xImageDisp.IsOfType<TPixelDisp>())
				{
					throw CLU_EXCEPTION("Given disparity image has incorrect type");
				}

				if (xImageL.IsOfType<Clu::TPixel_Lum_UInt8>()
					&& xImageR.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					_DoProcess<TPixelDisp, Clu::TPixel_Lum_UInt8>();
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}
			}

		} // namespace BlockMatcherSAD1
	} // namespace Cuda
} // namespace Clu



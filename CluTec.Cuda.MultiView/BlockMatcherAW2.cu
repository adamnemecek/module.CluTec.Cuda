////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherAW2.cu
//
// summary:   block matcher a w 2 class
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

#include "BlockMatcherAW2.h"

#include "CluTec.Math/Conversion.h"
#include "CluTec.Cuda.Base/PixelTypeInfo.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/Conversion.h"

#define DEBUG_BLOCK_X 40
#define DEBUG_BLOCK_Y 42
//#define CLU_DEBUG_CACHE
#define CLU_DEBUG_KERNEL

#include "CluTec.Cuda.Base/Kernel.Debug.h"
#include "CluTec.Cuda.Base/Kernel.ArrayCache.h"
#include "CluTec.Cuda.ImgProc/Kernel.Algo.AdaptWnd3D.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace BlockMatcherAW2
		{

			namespace Kernel
			{

				using namespace Clu::Cuda::Kernel;


				template<int t_iSubPatchSizeX, int t_iSubPatchSizeY
					, int t_iDispPerThread
					, int t_iWarpsPerBlockX, int t_iWarpsPerBlockY>
				struct Constants
				{
					// Warps per block
					static const int WarpsPerBlockX = t_iWarpsPerBlockX;
					static const int WarpsPerBlockY = t_iWarpsPerBlockY;

					// Thread per warp
					static const int ThreadsPerWarp = 32;
					static const int ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;

					// Number of disparities calculated per thread
					static const int DispPerThread = t_iDispPerThread;

					using AlgoAW = AlgoAdaptWnd3D::Impl<AlgoAdaptWnd3D::EType::Two_OutOf_Four, WarpsPerBlockX, t_iSubPatchSizeX, t_iSubPatchSizeY, DispPerThread>;

					// Used for calculating whether we can calculate disparity for a thread.
					using AlgoAW2D = AlgoAdaptWnd3D::Impl<AlgoAdaptWnd3D::EType::Two_OutOf_Four, WarpsPerBlockX, t_iSubPatchSizeX, t_iSubPatchSizeY, 2>;

					// Total number of disparities calculated per execution of AlgoAW
					static const int DispPerLoopCount = AlgoAW::PatchSizeZ;


					// the width of a base patch has to be a full number of words
					static const int BasePatchSizeX = AlgoAW::PatchSizeX;
					static const int BasePatchSizeY = AlgoAW::PatchSizeY;
					static const int BasePatchElementCount = AlgoAW::EffectivePatchElementCount;

					static const int BasePatchCountX = AlgoAW::PatchCountX;
					static const int BasePatchCountY = AlgoAW::PatchCountY;

					static const int TestBlockSizeX = AlgoAW::DataArraySizeX + 1;
					static const int TestBlockSizeY = AlgoAW::DataArraySizeY;

					static const int SumCacheSizeX = AlgoAW::SumCacheSizeX;
					static const int SumCacheSizeY = AlgoAW::SumCacheSizeY;

					static const int ResultsPerThread = AlgoAW::EvalLoopCount;



#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

					__device__ static void PrintStaticValues()
					{
						printf("Block Idx: %d, %d\n", blockIdx.x, blockIdx.y);
						AlgoAW::PrintStaticValues();
						AlgoAW::AlgoSum::PrintStaticValues();
						//PRINT();
						printf("\n");

						__syncthreads();
					}
#undef PRINT
				};


				__constant__ _SParameter c_xPars;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDisp;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDispInit;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgL;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgR;


				template<typename TResult, typename TValue>
				__device__ __forceinline__ TResult AbsoluteDifference(TValue xValA, TValue xValB)
				{
					return (TResult)abs(int(xValA) - int(xValB));
				}

				template<typename TResult, typename TValue>
				__device__ __forceinline__ TResult CensusDifference(TValue xValA, TValue xValB)
				{
					unsigned int uValue = ((unsigned int)xValA) ^ ((unsigned int)xValB);
					return (TResult)(__popc(uValue));
				}


				template<typename Const, typename TMinRange, typename TDispRange, typename TSum>
				__device__ __forceinline__ void UpdateDisp(TMinRange &xMinRange, TDispRange &xDisp, const TSum& iValue, int iDispIdx, int iDispIdxInc)
				{
					if (xDisp > TDisp(EDisparityId::Last))
					{
						return;
					}

					float fMin = float(iValue) / float(Const::BasePatchElementCount);

					// xMinRange.x: Minimum value
					// xMinRange.w: Value at previous disparity step
					// xMinRange.y: Value at disparity step just before minimum
					// xMinRange.z: Value at disparity step just after minimum
					if (iDispIdx == 0)
					{
						xMinRange.w = iValue;
						xMinRange.y = iValue;
						xMinRange.z = iValue;
					}

					//int iHasPrevMin = int(xMinRange.z > xMinRange.x && pxDisp[iResultIdx] + Const::BasePatchSizeX / 2 < iDispIdx);
					int iIsBelowMin = int(iValue < xMinRange.x && fMin < c_xPars.fSadThresh);
					//int iIsClearMin = int(iValue + 10 < xMinRange.x && fMin < fSadThresh);

					/*if (iIsClearMin && iHasPrevMin)
					{
					pxDisp[iResultIdx] = DisparityId::NotUnique;
					}
					else */if (iIsBelowMin)
					{
						// We have a new minimum, so set the value before minimum to previous value.
						xMinRange.y = xMinRange.w;

						// Set the minimum
						xMinRange.x = iValue;

						// Set the value after the minimum also to the minimum.
						// Can only be set properly in the next step.
						xMinRange.z = iValue;

						// Set disparity at minimum
						xDisp = iDispIdx;
					}
					else
					{
						// The current value is not a new minimum.
						// If the previous value is the minimum then set the value after the minimum to this value.
						if (xDisp + iDispIdxInc == iDispIdx)
						{
							xMinRange.z = iValue;
						}
					}

					// Set current value to previous value
					xMinRange.w = iValue;

				}


				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Eval graduated. </summary>
				///
				/// <typeparam name="Const">		  	Type of the constant. </typeparam>
				/// <typeparam name="TDisp">	  	Type of the disp component. </typeparam>
				/// <typeparam name="TCacheBasePatch">	Type of the cache base patch. </typeparam>
				/// <typeparam name="TCacheSum">	  	Type of the cache sum. </typeparam>
				/// <param name="pxDisp">		  	[in,out] If non-null, the px disp. </param>
				/// <param name="xCacheBasePatch">	[in,out] The cache base patch. </param>
				/// <param name="xCacheSumY">	  	[in,out] The cache sum y coordinate. </param>
				/// <param name="fSadThresh">	  	The sad thresh. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename Const, typename TDisp, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalGrad(TDisp pxDisp[Const::ResultsPerThread] 
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY
					, float fGradThresh)
				{
					using TSum = typename TCacheSum::TElement;

					Const::AlgoAW2D::EvalCache(xCacheSumY,
						[&xCacheBasePatch](int iIdxX, int iIdxY, int iIdxZ)
					{
						auto& pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						auto& pixVal2 = xCacheBasePatch.At(iIdxX + 1, iIdxY);
						return AbsoluteDifference<TSum>(pixVal1.x, pixVal2.x);
						//return CensusDifference<TSum>(pixVal1.x, pixVal2.x);
					},
						[&pxDisp, &fGradThresh](TSum& iValue, int iRelX, int iIdxX, int iIdxY, int iIdxZ, int iRelZ)
					{
						// Need only one result. TODO: Calculate gradient left and right of center.
						if (iIdxZ > 0)
						{
							return;
						}

						float fValue = float(iValue) / float(Const::BasePatchElementCount);

						if (pxDisp[iRelX] == TDisp(EDisparityId::Unknown) && fValue < fGradThresh)
						{
							pxDisp[iRelX] = TDisp(EDisparityId::CannotEvaluate);
						}
					});
				}


				__device__ __forceinline__ int MinDisp(int iDispA, int iDispB)
				{
					using TDisp = int;

					if (iDispA <= TDisp(EDisparityId::Last))
					{
						if (iDispB <= TDisp(EDisparityId::Last))
						{
							return min(iDispA, iDispB);
						}
					}
					else
					{
						if (iDispB <= TDisp(EDisparityId::Last))
						{
							return iDispB;
						}
					}

					return iDispA;
				}

				__device__ __forceinline__ int MaxDisp(int iDispA, int iDispB)
				{
					using TDisp = int;

					if (iDispA <= TDisp(EDisparityId::Last))
					{
						if (iDispB <= TDisp(EDisparityId::Last))
						{
							return max(iDispA, iDispB);
						}
					}
					else
					{
						if (iDispB <= TDisp(EDisparityId::Last))
						{
							return iDispB;
						}
					}

					return iDispA;
				}


				template<typename Const>
				__device__ __forceinline__ bool GetResultXY(int &iIdxX, int &iIdxY, int iResultIdx)
				{
					iIdxX = (iResultIdx * Const::ThreadsPerBlockX + threadIdx.x) % Const::BasePatchCountX;
					iIdxY = (iResultIdx * Const::ThreadsPerBlockX + threadIdx.x) / Const::BasePatchCountX;

					return iIdxY < Const::BasePatchCountY;
				}

				template<typename Const>
				__device__ __forceinline__ bool HasResult(int iResultIdx)
				{
					return ((iResultIdx * Const::ThreadsPerBlockX + threadIdx.x) / Const::BasePatchCountX) < Const::BasePatchCountY;
				}


				template<typename Const, typename TCache>
				__device__ __forceinline__ static typename TCache::TElement& ResultCacheValueAt(TCache& xCache, int iIdxX, int iIdxY, int iIdxZ)
				{
					return xCache.DataPointer()[Const::AlgoAW::ResultCacheIdx<TCache::StrideX, TCache::StrideY>(iIdxX, iIdxY, iIdxZ)];
				}


				template<typename Const, typename TDispTuple, typename TDisp, typename TMinRange, typename TDispRange>
				__device__ __forceinline__ void WriteDisparity(TMinRange pxMinRange[Const::ResultsPerThread]
					, TDispRange pxDisp[Const::ResultsPerThread]
					, int iBlockX, int iBlockY)
				{
					for (int iResultIdx = 0; iResultIdx < Const::ResultsPerThread; ++iResultIdx)
					{
						int iIdxX, iIdxY;
						if (!GetResultXY<Const>(iIdxX, iIdxY, iResultIdx))
						{
							continue;
						}

						TMinRange& xMinRange = pxMinRange[iResultIdx];

						float fMin = float(xMinRange.x) / float(Const::BasePatchElementCount);

						int iIsBelowMin = int(fMin < c_xPars.fSadThresh);
						int iDiff1 = abs(int(xMinRange.y) - int(xMinRange.x));
						int iDiff2 = abs(int(xMinRange.z) - int(xMinRange.x));

						//Debug::Run([&xMinRange]()
						//{
						//	if (Debug::IsBlock(40, 40))
						//	{
						//		printf("Min Range: %d, %d, %d, %d\n", xMinRange.x, xMinRange.y, xMinRange.z, xMinRange.w);
						//	}
						//});

						TDispTuple pixValue;


						if (pxDisp[iResultIdx] > TDisp(EDisparityId::Last))
						{
							pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDisp[iResultIdx], 0, 0, 0);
						}
						else if (iIsBelowMin)
						{
							int iIsDispSaturated = (pxDisp[iResultIdx] <= TDisp(EDisparityId::First) + 4 || pxDisp[iResultIdx] >= TDisp(EDisparityId::First) + c_xPars.iDispRange - 4);

							if (iDiff1 >= c_xPars.iMinDeltaThresh && iDiff2 >= c_xPars.iMinDeltaThresh && iIsDispSaturated == 0)
							{
								pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::First) + pxDisp[iResultIdx], xMinRange.x / Const::BasePatchElementCount, iDiff1, iDiff2);
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
							, iIdxX + iBlockX + Const::BasePatchSizeX / 2
							, iIdxY + iBlockY + Const::BasePatchSizeY / 2);

					}

				}


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
					using Const = Kernel::Constants<Config::SubPatchSizeX, Config::SubPatchSizeY, Config::DispPerThread, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoAW = typename Const::AlgoAW;

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
					//Debug::Run([]()
					//{
					//	if (Debug::IsThreadAndBlock(0, 0, 0, 0))
					//	{
					//		Const::PrintStaticValues();
					//	}
					//});

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// The Cache. 
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
					// Initialize the minimum SAD and Disparity arrays

					const int iScale = c_surfDisp.Format().iWidth / c_surfDispInit.Format().iWidth;

					Debug::Run([&iScale]()
					{
						if (Debug::IsThreadAndBlock(0, 0, 0, 0))
						{
							printf("Scale: %d / %d = %d\n", c_surfDisp.Format().iWidth, c_surfDispInit.Format().iWidth, iScale);
						}
					});

					TMinRange pxMinRange[Const::ResultsPerThread];
					TDispRange pxDisp[Const::ResultsPerThread];

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Read an image area into the cache
					xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
					__syncthreads();

#					pragma unroll
					for (int i = 0; i < Const::ResultsPerThread; ++i)
					{
						pxMinRange[i] = make_ushort4(0xFFFF, 0, 0, 0);
						pxDisp[i] = TDisp(EDisparityId::Unknown);
					}

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Evaluate block gradient in left image to identify patches that are distinguishable from their neighbors
					EvalGrad<Const>(pxDisp, xCacheBasePatch, xCacheSumY, c_xPars.fGradThresh);

					int iDoEvalDisp = 0;
					for (int i = 0; i < Const::ResultsPerThread; ++i)
					{
						if (!HasResult<Const>(i))
						{
							break;
						}

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

						for (int i = 0; i < Const::ResultsPerThread; ++i)
						{
							int iIdxX, iIdxY;
							if (GetResultXY<Const>(iIdxX, iIdxY, i))
							{
								c_surfDisp.Write2D(pixValue
									, iIdxX + iBlockX + Const::BasePatchSizeX / 2
									, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
							}
						}

						return;
					}

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Start disparity evaluation
					//static const int c_iInitialDispInc = 1;

					for (int iDispIdx = 0; iDispIdx < c_xPars.iDispRange; iDispIdx += Const::DispPerLoopCount)
					{
						//const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange + iDispIdx;
						const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange / 2 + iDispIdx;
						const int iTestY = iBlockY;

						if (c_surfImgR.Format().IsRectInside(iTestX, iTestY, Const::TestBlockSizeX + Const::DispPerLoopCount, Const::TestBlockSizeY))
						{
							Const::AlgoAW::EvalStoreCache(xCacheSumY,
								[&xCacheBasePatch, iTestX, iTestY](int iIdxX, int iIdxY, int iIdxZ)
							{
								auto& pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
								auto pixVal2 = c_surfImgR.ReadPixel2D<TPixelSrc>(iTestX + iIdxX + iIdxZ, iTestY + iIdxY);
								return AbsoluteDifference<TSum>(pixVal1.x, pixVal2.r());
								//return CensusDifference<TSum>(pixVal1.x, pixVal2.r());

							});

							//EvalAW<Const, TPixelSrc>(pxMinRange, pxDisp, xCacheBasePatch, xCacheSumY, c_surfImgR
							//	, iTestX, iTestY
							//	, iDispIdx, 1, c_xPars.fSadThresh);

							// Loop over all disparities calculated in parallel to update minima.
							for (int iResultIdx = 0; iResultIdx < Const::ResultsPerThread; ++iResultIdx)
							{
								TMinRange &xMinRange = pxMinRange[iResultIdx];
								TDispRange &xDisp = pxDisp[iResultIdx];

								int iIdxX, iIdxY;
								if (GetResultXY<Const>(iIdxY, iIdxY, iResultIdx))
								{
									for (int iIdxZ = 0; iIdxZ < Const::DispPerLoopCount; ++iIdxZ)
									{
										TSum& iValue = ResultCacheValueAt<Const>(xCacheSumY, iIdxX, iIdxY, iIdxZ);
										UpdateDisp<Const>(xMinRange, xDisp, iValue, iDispIdx + iIdxZ, 1);
									}
								}
							}

						}
					}

					__syncthreads();

					WriteDisparity<Const, TDispTuple, TDisp>(pxMinRange, pxDisp, iBlockX, iBlockY);

					__syncthreads();

				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Matcher use disparity. </summary>
				///
				/// <typeparam name="TPixelDisp">	Type of the pixel disp. </typeparam>
				/// <typeparam name="TPixelSrc"> 	Type of the pixel source. </typeparam>
				/// <typeparam name="t_eConfig"> 	Type of the configuration. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				//				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig>
				//				__global__ void MatcherUseDisparity()
				//				{
				//					using Config = SConfig<t_eConfig>;
				//					using Const = Kernel::Constants<Config::SubPatchSizeX, Config::SubPatchSizeY, Config::SubPatchCountY_Pow2, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;
				//
				//					using AlgoAW = typename Const::AlgoAW;
				//
				//					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;
				//					using TComponent = typename TPixelSrc::TData;
				//
				//					using TDisp = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
				//					using TDisp = typename TPixelDisp::TData;
				//
				//					using TSum = short;// unsigned short;
				//					using TMinRange = ushort4;
				//					using TDispRange = unsigned short;
				//
				//
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//
				//					// Get position of thread in left image
				//					const int iBlockX = c_xPars._iBlockOffsetX + blockIdx.x * Const::BasePatchCountX;
				//					const int iBlockY = c_xPars._iBlockOffsetY + blockIdx.y * Const::BasePatchCountY;
				//
				//					if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
				//					{
				//						return;
				//					}
				//
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ////////////////////////////////////////////////////////////////////////////////////////////////
				//					Debug::Run([]()
				//					{
				//						if (Debug::IsThreadAndBlock(0, 0, 0, 0))
				//						{
				//							Const::PrintStaticValues();
				//						}
				//					});
				//
				//					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// The Cache. Always a set of uints to speed up reading.
				//					__shared__ Clu::Cuda::Kernel::CArrayCache<TElement
				//						, Const::TestBlockSizeX, Const::TestBlockSizeY
				//						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8>
				//						xCacheBasePatch;
				//
				//					__shared__ Clu::Cuda::Kernel::CArrayCache<TSum
				//						, Const::SumCacheSizeX, Const::SumCacheSizeY
				//						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8>
				//						xCacheSumY;
				//
				//
				//
				//					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// Initialize the minimum SAD and Disparity arrays
				//
				//					const int iScale = c_surfDisp.Format().iWidth / c_surfDispInit.Format().iWidth;
				//
				//					Debug::Run([&iScale]()
				//					{
				//						if (Debug::IsThreadAndBlock(0, 0, 0, 0))
				//						{
				//							printf("Scale: %d / %d = %d\n", c_surfDisp.Format().iWidth, c_surfDispInit.Format().iWidth, iScale);
				//						}
				//					});
				//
				//					TMinRange pxMinRange[Const::ResultsPerThread];
				//					TDispRange pxDisp[Const::ResultsPerThread];
				//
				//					int iDispMin = 0x000FFFFF, iDispMax = 0;
				//
				//#					pragma unroll
				//					for (int i = 0; i < Const::ResultsPerThread; ++i)
				//					{
				//						pxMinRange[i] = make_ushort4(0xFFFF, 0, 0, 0);
				//						pxDisp[i] = DisparityId::Unknown;
				//
				//						int iIdxX, iIdxY;
				//						if (AlgoAW::GetResultXY(iIdxX, iIdxY, i))
				//						{
				//							TDisp pixDisp;
				//							TDispRange xDispValue;
				//
				//							const int iPosX = (iIdxX + iBlockX + Const::BasePatchSizeX / 2) / iScale;
				//							const int iPosY = (iIdxY + iBlockY + Const::BasePatchSizeY / 2) / iScale;
				//
				//							c_surfDispInit.Read2D(pixDisp, iPosX, iPosY);
				//							if (pixDisp.x <= DisparityId::Last)
				//							{
				//								xDispValue = iScale * pixDisp.x;
				//								iDispMin = min(iDispMin, int(xDispValue));
				//								iDispMax = max(iDispMax, int(xDispValue));
				//							}
				//							else
				//							{
				//								pxDisp[i] = pixDisp.x;
				//							}
				//
				//							//if (pixDisp.x <= DisparityId::Last)
				//							//{
				//							//	pxDispInit[i] = iScale * pixDisp.x;
				//							//}
				//							//else
				//							//{
				//							//	pxDispInit[i] = pixDisp.x;
				//							//}
				//
				//							//TDisp pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDispInit[i], 0, 0, 0);
				//							//c_surfDisp.Write2D(pixDisp
				//							//	, iIdxX + iBlockX + Const::BasePatchSizeX / 2
				//							//	, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
				//
				//						}
				//						//else
				//						//{
				//						//	pxDispInit[i] = DisparityId::Unknown;
				//						//}
				//						//__syncthreads();
				//					}
				//
				//
				//					__syncthreads();
				//
				//					iDispMin = MinDisp(iDispMin, __shfl_xor(iDispMin, 16));
				//					iDispMin = MinDisp(iDispMin, __shfl_xor(iDispMin, 8));
				//					iDispMin = MinDisp(iDispMin, __shfl_xor(iDispMin, 4));
				//					iDispMin = MinDisp(iDispMin, __shfl_xor(iDispMin, 2));
				//					iDispMin = MinDisp(iDispMin, __shfl_xor(iDispMin, 1));
				//
				//					iDispMax = MaxDisp(iDispMax, __shfl_xor(iDispMax, 16));
				//					iDispMax = MaxDisp(iDispMax, __shfl_xor(iDispMax, 8));
				//					iDispMax = MaxDisp(iDispMax, __shfl_xor(iDispMax, 4));
				//					iDispMax = MaxDisp(iDispMax, __shfl_xor(iDispMax, 2));
				//					iDispMax = MaxDisp(iDispMax, __shfl_xor(iDispMax, 1));
				//
				//					if (iDispMin > DisparityId::Last || iDispMax > DisparityId::Last)
				//					{
				//						iDispMin = DisparityId::Last + 1;
				//						iDispMax = 0;
				//					}
				//					else
				//					{
				//						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//						// Read an image area into the cache
				//						xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
				//						__syncthreads();
				//
				//					}
				//
				//
				//					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//					// Start disparity evaluation
				//					//static const int c_iInitialDispInc = 1;
				//
				//					for (int iDispIdx = (iDispMin - 4 < 0 ? 0 : iDispMin - 4); iDispIdx <= iDispMax + 4 /*c_xPars.iDispRange*/; ++iDispIdx)
				//					{
				//						//const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange + iDispIdx;
				//						const int iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange / 2 + iDispIdx;
				//						const int iTestY = iBlockY;
				//
				//						if (!c_surfImgR.Format().IsRectInside(iTestX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
				//						{
				//							continue;
				//						}
				//
				//						EvalAW<Const, TPixelSrc>(pxMinRange, pxDisp, xCacheBasePatch, xCacheSumY, c_surfImgR
				//							, iTestX, iTestY
				//							, iDispIdx, 1, c_xPars.fSadThresh);
				//
				//					}
				//
				//					__syncthreads();
				//
				//					WriteDisparity<t_eConfig, TDisp, TDisp>(pxMinRange, pxDisp, iBlockX, iBlockY);
				//
				//					__syncthreads();
				//
				//				}


			} // namespace Kernel



			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Executes the configure operation. </summary>
			///
			/// <typeparam name="t_eConfig">	Type of the configuration. </typeparam>
			/// <param name="xDevice">	The device. </param>
			/// <param name="xFormat">	Describes the format to use. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<EConfig t_eConfig>
			void CDriver::_DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				using Config = SConfig<t_eConfig>;
				using Const = Kernel::Constants<Config::SubPatchSizeX, Config::SubPatchSizeY, Config::DispPerThread, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

				//unsigned nByteOffset = 0; // (m_xPars.iOffset) * (unsigned)xFormat.BytesPerPixel();
				unsigned nOffsetLeft = 0; // ((nByteOffset / 128 + (nByteOffset % 128 > 0 ? 1 : 0)) * 128) / (unsigned)xFormat.BytesPerPixel();

				unsigned nOffsetRight = Const::TestBlockSizeX - Const::BasePatchCountX;

				unsigned nOffsetTop = 2;
				unsigned nOffsetBottom = Const::TestBlockSizeY - Const::BasePatchCountY;

				m_xPars._iBlockOffsetX = 0; // (int)nOffsetLeft;
				m_xPars._iBlockOffsetY = 0;
				m_xPars._iRightToLeftOffsetX = m_xPars.iOffset; // (int)nOffsetLeft;

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
				case EConfig::Patch_15x15:
					_DoConfigure<EConfig::Patch_15x15>(xDevice, xFormat);
					break;

				case EConfig::Patch_9x9:
					_DoConfigure<EConfig::Patch_9x9>(xDevice, xFormat);
					break;

				default:
					throw CLU_EXCEPTION("Invalid algorithm configuration.");
				}

			}

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Executes the process operation. </summary>
			///
			/// <typeparam name="TPixelDisp">	Type of the pixel disp. </typeparam>
			/// <typeparam name="TPixelSrc"> 	Type of the pixel source. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<typename TPixelDisp, typename TPixelSrc>
			void CDriver::_DoProcess()
			{
				if (m_xPars.iUseDispInput)
				{
					//switch (m_xPars.eConfig)
					//{
					//case EConfig::Patch_15x15:
					//	Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, EConfig::Patch_15x15>
					//		CLU_KERNEL_CONFIG()();
					//	break;

					//case EConfig::Patch_9x9:
					//	Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, EConfig::Patch_9x9>
					//		CLU_KERNEL_CONFIG()();
					//	break;

					//default:
					//	throw CLU_EXCEPTION("Invalid algorithm configuration.");
					//}
				}
				else
				{
					switch (m_xPars.eConfig)
					{
					case EConfig::Patch_15x15:
						Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_15x15>
							CLU_KERNEL_CONFIG()();
						break;

					case EConfig::Patch_9x9:
						Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_9x9>
							CLU_KERNEL_CONFIG()();
						break;

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}
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

			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageDisp
				, const Clu::Cuda::_CDeviceSurface& xImageL
				, const Clu::Cuda::_CDeviceSurface& xImageR
				, const Clu::Cuda::_CDeviceSurface& xImageDispInit)
			{
				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDisp, &xImageDisp, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDispInit, &xImageDispInit, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
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
		} // namespace BlockMatcherAW1
	} // namespace Cuda

} // namespace Clu
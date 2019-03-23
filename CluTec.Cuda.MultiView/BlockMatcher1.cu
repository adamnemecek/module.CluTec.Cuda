////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcher1.cu
//
// summary:   block matcher 1 class
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
#include "cuda_fp16.h"

#include "CluTec.Types1/Pixel.h"

#include "BlockMatcher1.h"

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
#include "CluTec.Cuda.Base/Kernel.ArrayCache_W16.h"
#include "CluTec.Cuda.ImgProc/Kernel.Algo.ArraySum.h"
//#include "CluTec.Cuda.ImgProc/Kernel.Algo.ArraySum_W32H16.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace BlockMatcher1
		{

			namespace Kernel
			{

				using namespace Clu::Cuda::Kernel;

				using TIdx = short;

				template<int t_iPatchCountY_Pow2, int t_iPatchSizeX, int t_iPatchSizeY
					, int t_iWarpsPerBlockX, int t_iWarpsPerBlockY>
					struct Constants
				{
					// Warps per block
					static const TIdx WarpsPerBlockX = t_iWarpsPerBlockX;
					static const TIdx WarpsPerBlockY = t_iWarpsPerBlockY;

					// Thread per warp
					static const TIdx ThreadsPerWarp = 32;
					static const TIdx ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;

					using AlgoSum = Clu::Cuda::Kernel::AlgoArraySum<ThreadsPerBlockX, t_iPatchCountY_Pow2, t_iPatchSizeX, t_iPatchSizeY>;
					//using AlgoSum = Clu::Cuda::Kernel::AlgoArraySum_W32H16<ThreadsPerBlockX, t_iPatchSizeX, t_iPatchSizeY>;



					// the width of a base patch has to be a full number of words
					static const TIdx BasePatchSizeX = AlgoSum::PatchSizeX;
					static const TIdx BasePatchSizeY = AlgoSum::PatchSizeY;
					static const TIdx BasePatchElementCount = AlgoSum::PatchElementCount;

					static const TIdx BasePatchCountX = AlgoSum::PatchCountX;
					static const TIdx BasePatchCountY = AlgoSum::PatchCountY;

					static const TIdx TestBlockSizeX = AlgoSum::DataArraySizeX + 1;
					static const TIdx TestBlockSizeY = AlgoSum::DataArraySizeY;

					static const TIdx SumCacheSizeX = AlgoSum::SumCacheSizeX;
					static const TIdx SumCacheSizeY = AlgoSum::SumCacheSizeY;

					static const TIdx ResultCountPerThread = AlgoSum::ColGroupSizeX;

					static const TIdx SubDispCount = 16;

#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

					__device__ static void PrintStaticValues()
					{
						printf("Block Idx: %d, %d\n", blockIdx.x, blockIdx.y);
						AlgoSum::PrintStaticValues();
						//PRINT();
						printf("\n");

						__syncthreads();
					}
#undef PRINT
				};

				template<typename T>
				struct SSimStore
				{
					using TValue = T;

					__device__ __forceinline__ void SetZero()
					{
						iPrev2 = iPrev = iMax = iNext = iNext2 = iLast = T(0);
					}

					T iPrev2;
					T iPrev;
					T iMax;
					T iNext;
					T iNext2;
					T iLast;
				};


				__constant__ _SParameter c_xPars;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDisp;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDispInit;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgL;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgR;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDebug;

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Simulation value to float. </summary>
				///
				/// <typeparam name="Const">		Type of the constant. </typeparam>
				/// <typeparam name="TSrcValue">	Type of the source value. </typeparam>
				/// <typeparam name="TSimValue">	Type of the simulation value. </typeparam>
				/// <param name="iSimValue">	Zero-based index of the simulation value. </param>
				///
				/// <returns>	A float. </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename Const, typename TSrcValue, typename TSimValue>
				__device__ __forceinline__ float SimValueToFloat(const TSimValue &iSimValue)
				{
					return (float(iSimValue) / float(Const::BasePatchElementCount)) * (100.0f / float(Clu::NumericLimits<TSrcValue>::Max() >> 1));
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>
				/// Similarity float to Integer value. Assumes the similarity value lies in the range [-1, 1].
				/// </summary>
				///
				/// <typeparam name="TResult">	Type of the disp. </typeparam>
				/// <param name="fSimValue">	[in,out] The simulation value. </param>
				///
				/// <returns>	A TDisp. </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TResult, typename TSrcValue>
				__device__ __forceinline__ TResult SimFloatToInteger(const float &fSimValue)
				{
					return (TResult)(Clu::Clamp((fSimValue / 100.0f + 1.0f) / 2.0f, 0.0f, 1.0f) * float(Clu::NumericLimits<TResult>::Max()));
				}


				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Absolute difference similarity. Evaluates a similarity value based on the 
				/// 			absolute difference of the input values. The maximal value of TValue
				/// 			represents the highest similarity.</summary>
				///
				/// <typeparam name="TResult">	Type of the result. </typeparam>
				/// <typeparam name="TValue"> 	Type of the value. </typeparam>
				/// <param name="xValA">	The value a. </param>
				/// <param name="xValB">	The value b. </param>
				///
				/// <returns>	A TResult. </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////
				template<typename TResult, typename TValue>
				__device__ __forceinline__ static TResult EvalSim(TValue xValA, TValue xValB, SSimType_AbsoluteDifference)
				{
					return (TResult)(TResult(Clu::NumericLimits<TValue>::Max()) - abs(TResult(xValA) - TResult(xValB)) >> 1);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Signed product. This function assumes that the input values are signed values coded
				/// 			in an unsigned type by adding half the types value range to the original values.
				/// 			The output is a signed value. That is, TResult has to be a signed type and must have 
				/// 			more bits than the TValue type.
				/// 			
				/// 			 </summary>
				///
				/// <typeparam name="TResult">	Type of the result. </typeparam>
				/// <typeparam name="TValue"> 	Type of the value. </typeparam>
				/// <param name="xValA">	The value a. </param>
				/// <param name="xValB">	The value b. </param>
				///
				/// <returns>	A TResult. </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TResult, typename TValue>
				__device__ __forceinline__ static TResult EvalSim(TValue xValA, TValue xValB, SSimType_SignedProduct)
				{
					float fVal1 = 2.0f * (float(xValA) / float(Clu::NumericLimits<TValue>::Max()) - 0.5f);
					float fVal2 = 2.0f * (float(xValB) / float(Clu::NumericLimits<TValue>::Max()) - 0.5f);
					fVal1 *= fVal2;
					fVal1 = fVal1 * float(Clu::NumericLimits<TValue>::Max() >> 1);

					return TResult(fVal1);
				}


				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Eval sad. </summary>
				///
				/// <typeparam name="Const">		  	Type of the constant. </typeparam>
				/// <typeparam name="TPixelSurf">	  	Type of the pixel surf. </typeparam>
				/// <typeparam name="TSimStore">	  	Type of the minimum range. </typeparam>
				/// <typeparam name="TDisp">	  	Type of the disp component. </typeparam>
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

				template<typename Const, typename TSimType, typename TPixelSurf, typename TSimStore
					, typename TDisp
					, typename TCacheBasePatch
					, typename TCacheSum, typename TCacheTestPatch>
					__device__ void EvalSim(TSimStore* pxSimStore
						, TDisp* pxDisp
						, TCacheBasePatch& xCacheBasePatch
						, TCacheSum& xCacheSumY
						, TCacheTestPatch& xCacheTestPatch
						, TIdx iBlockX, TIdx iBlockY, TDisp iDispIdx, const TDisp uDispIdxInc)
				{
					using TSum = typename TCacheSum::TElement;
					using TSimValue = typename TSimStore::TValue;
					
					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch, &xCacheTestPatch, iBlockX, iBlockY](TIdx iIdxX, TIdx iIdxY)
					{
						auto pixValBase = xCacheBasePatch.At(iIdxX, iIdxY);
						auto pixValTest = xCacheTestPatch.At(iBlockX + iIdxX, iBlockY + iIdxY);
						return EvalSim<TSum>(pixValBase.x, pixValTest.x, TSimType());
					},
						[&](TSum& iSum, TIdx iResultIdx, TIdx iIdxX, TIdx iIdxY)
					{
						if (pxDisp[iResultIdx] > TDisp(EDisparityId::Last))
						{
							return;
						}

						auto xSimStore = pxSimStore[iResultIdx];

						const TSimValue iValue = TSimValue(iSum);

						const TSimValue iIsAboveMax = TSimValue(iValue > xSimStore.iMax);
						const TSimValue iIsJustBehindMin = (1 - iIsAboveMax)
							* TSimValue(pxDisp[iResultIdx] + uDispIdxInc == (iDispIdx + TDisp(EDisparityId::First)));
						const TSimValue iIsJust2BehindMin = (1 - iIsAboveMax)
							* TSimValue(pxDisp[iResultIdx] + 2 * uDispIdxInc == (iDispIdx + TDisp(EDisparityId::First)));

						xSimStore.iPrev2 = (1 - iIsAboveMax) * xSimStore.iPrev2 + iIsAboveMax * xSimStore.iPrev;

						// We have a new minimum, so set the value before minimum to previous value.
						xSimStore.iPrev = (1 - iIsAboveMax) * xSimStore.iPrev + iIsAboveMax * xSimStore.iLast;

						// Set the minimum
						xSimStore.iMax = (1 - iIsAboveMax) * xSimStore.iMax + iIsAboveMax * iValue;

						// Set the value after the minimum also to the minimum.
						// Can only be set properly in the next step.
						xSimStore.iNext = (1 - iIsAboveMax) * xSimStore.iNext + iIsAboveMax * iValue;
						xSimStore.iNext2 = (1 - iIsAboveMax) * xSimStore.iNext2 + iIsAboveMax * iValue;

						// Set disparity at minimum
						pxDisp[iResultIdx] = TDisp(1 - iIsAboveMax) * pxDisp[iResultIdx] + TDisp(iIsAboveMax) * (iDispIdx + TDisp(EDisparityId::First));

						// The current value is not a new minimum.
						// If the previous value is the minimum then set the value after the minimum to this value.
						xSimStore.iNext = (1 - iIsJustBehindMin) * xSimStore.iNext + iIsJustBehindMin * iValue;
						xSimStore.iNext2 = (1 - iIsJust2BehindMin) * xSimStore.iNext2 + iIsJust2BehindMin * iValue;


						// Set current value to previous value
						xSimStore.iLast = iValue;

						pxSimStore[iResultIdx] = xSimStore;

					});


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

				template<typename Const, typename TSrcValue, typename TDisp, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalGrad(TDisp* pxDisp
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY
					, float fGradThresh)
				{
					using TSum = typename TCacheSum::TElement;

					TSum piSum[Const::ResultCountPerThread];
					
					Const::AlgoSum::SumCache(xCacheSumY,
						[&](TIdx iIdxX, TIdx iIdxY)
					{
						return (TSum)xCacheBasePatch.At(iIdxX, iIdxY).x;
					},
						[&](TSum& iValue, TIdx iRelX, TIdx iIdxX, TIdx iIdxY)
					{
						piSum[iRelX] = iValue;
					});

					Const::AlgoSum::SumCache(xCacheSumY,
						[&](TIdx iIdxX, TIdx iIdxY)
					{
						TSum iValue = (TSum)xCacheBasePatch.At(iIdxX, iIdxY).x;
						return (iValue * iValue);
					},
						[&](TSum& iSum2, TIdx iRelX, TIdx iIdxX, TIdx iIdxY)
					{
						const float fMean = float(piSum[iRelX]) / float(Const::BasePatchElementCount);
						const float fMean2 = float(iSum2) / float(Const::BasePatchElementCount);
						const float fStdDev = sqrtf(abs(fMean2 - fMean * fMean)) / float(Clu::NumericLimits<TSrcValue>::Max()) * 100.0f;

						if (pxDisp[iRelX] == TDisp(EDisparityId::Unknown) && fStdDev < fGradThresh)
						{
							pxDisp[iRelX] = TDisp(EDisparityId::CannotEvaluate);
						}
					});
				}


				template<typename TDisp>
				__device__ __forceinline__ int IsDispValid(TDisp uDisp)
				{
					return (uDisp >= TDisp(EDisparityId::First) && uDisp <= TDisp(EDisparityId::Last));
				}

				template<typename TDisp>
				__device__ __forceinline__ TDisp MinDisp(TDisp uDispA, TDisp uDispB)
				{
					if (IsDispValid(uDispA))
					{
						if (IsDispValid(uDispB))
						{
							return min(uDispA, uDispB);
						}
					}
					else
					{
						if (IsDispValid(uDispB))
						{
							return uDispB;
						}
					}

					return uDispA;
				}

				template<typename TDisp>
				__device__ __forceinline__ TDisp MaxDisp(TDisp uDispA, TDisp uDispB)
				{
					if (IsDispValid(uDispA))
					{
						if (IsDispValid(uDispB))
						{
							return max(uDispA, uDispB);
						}
					}
					else
					{
						if (IsDispValid(uDispB))
						{
							return uDispB;
						}
					}

					return uDispA;
				}

				__device__ __forceinline__ ushort2 MinMaxDisp(ushort2 uDispA, ushort2 uDispB)
				{
					ushort2 uResult;

					uResult.x = MinDisp(uDispA.x, uDispB.x);
					uResult.y = MaxDisp(uDispA.y, uDispB.y);

					return uResult;
				}


				template<typename TValue>
				__device__ __forceinline__ TValue ShuffleXor(TValue tValue, int iLaneMask, int iWidth = 32)
				{
					static_assert(sizeof(tValue) == 4, "Given value is not a 32bit type");

					int iValue = *((int*)&tValue);
					int iResult;
					TValue tResult;

					iResult = __shfl_xor(iValue, iLaneMask, iWidth);

					tResult = *((TValue*)&iResult);
					return tResult;
				}


				template<EConfig t_eConfig, typename TDispTuple, typename TDisp, typename TSrcValue, typename TSimStore>
				__device__ __forceinline__ void WriteDisparity(TSimStore* pxSimStore, TDisp* pxDisp, TIdx iBlockX, TIdx iBlockY)
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchCountY_Pow2, Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;
					using AlgoSum = typename Const::AlgoSum;


					for (TIdx iResultIdx = 0; iResultIdx < Const::ResultCountPerThread; ++iResultIdx)
					{
						TIdx iIdxX, iIdxY;
						if (AlgoSum::GetResultXY(iIdxX, iIdxY, iResultIdx))
						{
							//if (Debug::IsBlock(5, 5))
							//{
							//	printf("%d: %d, %d\n", iResultIdx, iIdxX, iIdxY);
							//}

							// Ensure that a whole Base Patch is inside the image to write a result
							if (c_surfDisp.IsInside(iIdxX + iBlockX + Const::BasePatchSizeX, iIdxY + iBlockY + Const::BasePatchSizeY))
							{
								TDispTuple pixValue;

								if (pxDisp[iResultIdx] > TDisp(EDisparityId::Last))
								{
									pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDisp[iResultIdx], 0, 0, 0);
								}
								else if (pxDisp[iResultIdx] == TDisp(EDisparityId::Unknown))
								{
									pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::Unknown), 0, 0, 0);
								}
								else
								{
									TSimStore& xSimStore = pxSimStore[iResultIdx];

									TIdx iIsSpecific, iIsAboveMax;
									TDisp uSimMax, uSimDiffPrev, uSimDiffNext;

									{
										const float fSimMax = SimValueToFloat<Const, TSrcValue>(xSimStore.iMax);
										const float fSimPrev2 = SimValueToFloat<Const, TSrcValue>(xSimStore.iPrev2);
										const float fSimPrev = SimValueToFloat<Const, TSrcValue>(xSimStore.iPrev);
										const float fSimNext = SimValueToFloat<Const, TSrcValue>(xSimStore.iNext);
										const float fSimNext2 = SimValueToFloat<Const, TSrcValue>(xSimStore.iNext2);
										const float fDiffPrev = max(abs(fSimPrev - fSimMax), abs(fSimPrev2 - fSimMax));
										const float fDiffNext = max(abs(fSimNext - fSimMax), abs(fSimNext2 - fSimMax));

										iIsSpecific = TIdx(fDiffPrev >= c_xPars.fMinDeltaThresh && fDiffNext >= c_xPars.fMinDeltaThresh);
										iIsAboveMax = TIdx(fSimMax >= c_xPars.fSimThresh);

										uSimMax = SimFloatToInteger<TDisp, TSrcValue>(fSimMax);
										uSimDiffPrev = SimFloatToInteger<TDisp, TSrcValue>(fDiffPrev);
										uSimDiffNext = SimFloatToInteger<TDisp, TSrcValue>(fDiffNext);

										//if (blockIdx.x == 12 && blockIdx.y == 10 && threadIdx.x == 26) // && iResultIdx == 0)
										//{
										//	printf("Thread: %d/%d, Disp: %d | SAD [%d -> %d -> %d -> %d -> %d]; [%g -> %g -> %g -> %g -> %g]\n"
										//		, threadIdx.x, iResultIdx, pxDisp[iResultIdx]
										//		, xSimStore.iPrev2, xSimStore.iPrev, xSimStore.iMax, xSimStore.iNext, xSimStore.iNext2
										//		, fSimPrev2, fSimPrev, fSimMax, fSimNext, fSimNext2);
										//}
									}

									//Debug::Run([&]()
									//{
									//	if (Debug::IsBlock(12, 10))
									//	{
									//		printf("Disp > Min Range: %d: %d > %d, %d, %d, %d\n", threadIdx.x, pxDisp[iResultIdx], xMinRange.x, xMinRange.y, xMinRange.z, xMinRange.w);
									//	}
									//});

									//if (blockIdx.x == 12 && blockIdx.y == 10 && threadIdx.x == 26 && iResultIdx == 0)
									//{
									//	printf("Thread: %d, Disp: %d | SAD [%d -> %d -> %d -> %d -> %d], prev: %d, Diffs: %d, %d | %d\n"
									//		, threadIdx.x, pxDisp[iResultIdx]
									//		, xMinRange.uPrev2, xMinRange.uPrev, xMinRange.uMin, xMinRange.uNext, xMinRange.uNext2, xMinRange.uLast
									//		, iDiff1, iDiff2, c_xPars.iMinDeltaThresh);
									//}

									//pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(xMinRange.x, xMinRange.x, iDiff1, iDiff2);


									if (iIsAboveMax)
									{
										const TIdx iIsDispSaturated = (pxDisp[iResultIdx] <= TDisp(EDisparityId::First) + TDisp(c_xPars.xDispConfig.Min()) + 1
											|| pxDisp[iResultIdx] >= TDisp(EDisparityId::First) + c_xPars.xDispConfig.Max() - 1);


										if (iIsDispSaturated)
										{
											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::Saturated), uSimMax, uSimDiffPrev, uSimDiffNext);
										}
										else if (iIsSpecific)
										{
											//if (iIdxY + iBlockY == 646)
											//{
											//	printf("%d, %d, %d, %d: %d, %d, %d, %d, %g\n", iIdxX + iBlockX, iIdxY + iBlockY, threadIdx.x, iResultIdx
											//		, pxDisp[iResultIdx], uSimMax, uSimDiffPrev, uSimDiffNext, fSimMax);
											//}

											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDisp[iResultIdx], uSimMax, uSimDiffPrev, uSimDiffNext);
										}
										else
										{
											//if (blockIdx.x == 12 && blockIdx.y == 10)
											//{
											//	printf("Not Specific: %d [%d]\n", threadIdx.x, iResultIdx);
											//}

											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotSpecific), uSimMax, uSimDiffPrev, uSimDiffNext);
										}
									}
									else
									{
										pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotFound), uSimMax, uSimDiffPrev, uSimDiffNext);
									}
								}

								c_surfDisp.Write2D(pixValue
									, iIdxX + iBlockX + Const::BasePatchSizeX / 2
									, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
							}
						}

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

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig, typename TSimType>
				__global__ void Matcher()
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchCountY_Pow2, Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoSum = typename Const::AlgoSum;

					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;

					using TComponent = typename TPixelSrc::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TSum = int;
					using TSimValue = short;
					using TSimStore = SSimStore<TSimValue>;
					using TFloat = float;

					using TCachePatch = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheTest = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX + Const::SubDispCount - 1, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheSum = Clu::Cuda::Kernel::CArrayCache<TSum
						, Const::SumCacheSizeX, Const::SumCacheSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////

					// Get position of thread in left image
					const TIdx iBlockX = TIdx(c_xPars._iBlockOffsetX + blockIdx.x * Const::BasePatchCountX);
					const TIdx iBlockY = TIdx(c_xPars._iBlockOffsetY + blockIdx.y * Const::BasePatchCountY);

					//if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
					if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::BasePatchSizeX, Const::BasePatchSizeY))
					{
						return;
					}

					//if (!(blockIdx.x >= 12 && blockIdx.x <= 12))// && blockIdx.y == 10))
					//{
					//	return;
					//}

					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					//Debug::Run([]()
					//{
					//	if (Debug::IsThreadAndBlock(0, 0, 0, 0))
					//	{
					//		Const::PrintStaticValues();
					//		TCacheSum::PrintStaticValues();
					//	}
					//});

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// The Cache. Always a set of uints to speed up reading.
					__shared__ TCachePatch xCacheBasePatch;
					__shared__ TCacheTest xCacheTestPatch;
					__shared__ TCacheSum xCacheSumY;



					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Initialize the minimum SAD and Disparity arrays

					TSimStore pxSimStore[Const::ResultCountPerThread];
					TDisp pxDisp[Const::ResultCountPerThread];

#					pragma unroll
					for (TIdx i = 0; i < Const::ResultCountPerThread; ++i)
					{
						//pxMinRange[i] = make_ushort4(0xFFFF, 0, 0, 0);
						pxSimStore[i].SetZero();
						pxSimStore[i].iMax = -Clu::NumericLimits<TSimValue>::Max();
						//pxDispInit[i] = DisparityId::Unknown;
						pxDisp[i] = TDisp(EDisparityId::Unknown);
					}

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Read an image area into the cache
					if (c_surfImgL.IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
					{
						xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
					}
					else
					{
						xCacheBasePatch.SafeReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
					}
					__syncthreads();

					//Debug::Run([&]()
					//{
					//	xCacheBasePatch.WriteToSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfDebug, iBlockX, iBlockY
					//		, [&](TElement& xValue, Clu::Cuda::_CDeviceSurface& surfImg, int iImgX_px, int iImgY_px, int iCacheX_px, int iCacheY_px)
					//	{
					//		surfImg.Write2D<TElement>(xValue, iImgX_px, iImgY_px);
					//	});
					//	__syncthreads();
					//});

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Evaluate block gradient in left image to identify patches that are distinguishable from their neighbors
					EvalGrad<Const, TComponent>(pxDisp, xCacheBasePatch, xCacheSumY, c_xPars.fGradThresh);
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

					TIdx iDoEvalDisp = 0;
					for (TIdx i = 0; i < Const::ResultCountPerThread; ++i)
					{
						if (!AlgoSum::HasResult(i))
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

						for (TIdx i = 0; i < Const::ResultCountPerThread; ++i)
						{
							TIdx iIdxX, iIdxY;
							if (AlgoSum::GetResultXY(iIdxX, iIdxY, i))
							{
								// Ensure that a whole Base Patch is inside the image to write a result
								if (c_surfDisp.IsInside(iIdxX + iBlockX + Const::BasePatchSizeX, iIdxY + iBlockY + Const::BasePatchSizeY))
								{
									c_surfDisp.Write2D(pixValue
										, iIdxX + iBlockX + Const::BasePatchSizeX / 2
										, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
								}
							}
						}

						return;
					}


					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Start disparity evaluation
					//static const int c_iInitialDispInc = 1;

					//const int iMaxDispIdx = c_xPars.xDispConfig.Range();
					const int iMaxDispIdx = min(c_xPars.xDispConfig.Max()
						, c_xPars.xDispConfig.MaxDisparity(iBlockX, c_surfImgL.Format().iWidth, c_xPars._iIsLeftToRight));

					for (TIdx iDispIdx = c_xPars.xDispConfig.Min(); iDispIdx < iMaxDispIdx; iDispIdx += Const::SubDispCount)
					{
						const TIdx iTestX = c_xPars.xDispConfig.MapPixelPos(iBlockX, iDispIdx, c_xPars._iIsLeftToRight)
							- c_xPars._iIsLeftToRight * (Const::SubDispCount - 1);

						const TIdx iTestY = iBlockY;

						if (c_surfImgR.Format().IsRectInside(iTestX, iTestY, Const::TestBlockSizeX + Const::SubDispCount - 1, Const::TestBlockSizeY))
						{
							xCacheTestPatch.ReadFromSurf<Const::TestBlockSizeX + Const::SubDispCount - 1, Const::TestBlockSizeY>(c_surfImgR, iTestX, iTestY);
							__syncthreads();

							//Debug::Run([&]()
							//{
							//	xCacheTestPatch.WriteToSurf<Const::TestBlockSizeX + Const::SubDispCount, Const::TestBlockSizeY>(c_surfDebug, iBlockX, iBlockY + Const::TestBlockSizeY + 5
							//		, [&](TElement& xValue, Clu::Cuda::_CDeviceSurface& surfImg, int iImgX_px, int iImgY_px, int iCacheX_px, int iCacheY_px)
							//	{
							//		surfImg.Write2D<TElement>(xValue, iImgX_px, iImgY_px);
							//	});

							//	__syncthreads();
							//});


							for (TIdx iRelDisp = 0; iRelDisp < Const::SubDispCount; ++iRelDisp)
							{
								EvalSim<Const, TSimType, TPixelSrc>(pxSimStore, pxDisp
									, xCacheBasePatch, xCacheSumY, xCacheTestPatch
									, TIdx(c_xPars._iIsLeftToRight * (Const::SubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
									, TDisp(iDispIdx + iRelDisp), TDisp(1));
							}
						}
						// If not a whole SubDispCount block fits into the image, try to use a smaller block
						else
						{
							const _SImageFormat &xF = c_surfImgR.Format();

							if (iTestY >= 0 && iTestY + Const::TestBlockSizeY <= xF.iHeight)
							{
								int iDeltaX, iStartX;

								if (iTestX >= 0)
								{
									iDeltaX = xF.iWidth - iTestX - Const::TestBlockSizeX;
									iStartX = iTestX;

									//if (threadIdx.x == 0)
									//{
									//	printf("%d, %d > (%d) : %d - %d - %d = %d -> Start: %d\n"
									//		, blockIdx.x, blockIdx.y, iDispIdx, xF.iWidth, iTestX, Const::TestBlockSizeX, iDeltaX, iStartX);
									//}
								}
								else
								{
									iDeltaX = iTestX + (Const::SubDispCount - 1);
									iStartX = 0;

									//if (threadIdx.x == 0)
									//{
									//	printf("%d, %d > (%d) : %d + (%d - 1) = %d -> Start: %d\n"
									//		, blockIdx.x, blockIdx.y, iDispIdx, iTestX, Const::SubDispCount - 1, iDeltaX, iStartX);
									//}
								}

								if (iDeltaX >= 0)
								{
									const int iSubDispCount = iDeltaX + 1;

									xCacheTestPatch.ReadFromSurfVarWidth<Const::TestBlockSizeY>(c_surfImgR, iStartX, iTestY, Const::TestBlockSizeX + iSubDispCount - 1);
									__syncthreads();

									for (TIdx iRelDisp = 0; iRelDisp < iSubDispCount; ++iRelDisp)
									{
										EvalSim<Const, TSimType, TPixelSrc>(pxSimStore, pxDisp
											, xCacheBasePatch, xCacheSumY, xCacheTestPatch
											, TIdx(c_xPars._iIsLeftToRight * (iSubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
											, TDisp(iDispIdx + iRelDisp), TDisp(1));
									}

								}
							}
						}
					}

					__syncthreads();

					WriteDisparity<t_eConfig, TDispTuple, TDisp, TComponent>(pxSimStore, pxDisp, iBlockX, iBlockY);

					__syncthreads();

				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Matcher use disparity. </summary>
				///
				/// <typeparam name="TPixelDisp">	Type of the pixel disp. </typeparam>
				/// <typeparam name="TPixelSrc"> 	Type of the pixel source. </typeparam>
				/// <typeparam name="t_eConfig"> 	Type of the configuration. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig, typename TSimType>
				__global__ void MatcherUseDisparity()
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchCountY_Pow2, Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoSum = typename Const::AlgoSum;

					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;
					using TComponent = typename TPixelSrc::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TSum = int;
					using TSimValue = short;
					using TSimStore = SSimStore<TSimValue>;
					using TIdx = short;
					using TDispMinMax = ushort2;
					using TFloat = float;

					using TCachePatch = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheTest = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX + Const::SubDispCount - 1, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheSum = Clu::Cuda::Kernel::CArrayCache<TSum
						, Const::SumCacheSizeX, Const::SumCacheSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;


					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////

					//if (!Debug::IsBlock(200, 50))
					//{
					//	return;
					//}

					// Get position of thread in left image
					const TIdx iBlockX = TIdx(c_xPars._iBlockOffsetX + blockIdx.x * Const::BasePatchCountX);
					const TIdx iBlockY = TIdx(c_xPars._iBlockOffsetY + blockIdx.y * Const::BasePatchCountY);

					//if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
					if (!c_surfImgL.Format().IsRectInside(iBlockX, iBlockY, Const::BasePatchSizeX, Const::BasePatchSizeY))
					{
						return;
					}

					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					// ////////////////////////////////////////////////////////////////////////////////////////////////
					//Debug::Run([]()
					//{
					//	if (Debug::IsThreadAndBlock(0, 0, 20, 325))
					//	{
					//		Const::PrintStaticValues();
					//	}
					//});

					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// The Cache. Always a set of uints to speed up reading.
					__shared__ TCachePatch xCacheBasePatch;
					__shared__ TCacheTest xCacheTestPatch;
					__shared__ TCacheSum xCacheSumY;



					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Initialize the minimum SAD and Disparity arrays

					const TIdx iScale = c_surfDisp.Format().iWidth / c_surfDispInit.Format().iWidth;

					//Debug::Run([&iScale]()
					//{
					//	if (Debug::IsThreadAndBlock(0, 0, 0, 0))
					//	{
					//		printf("Scale: %d / %d = %d\n", c_surfDisp.Format().iWidth, c_surfDispInit.Format().iWidth, iScale);
					//	}
					//});

					TSimStore pxSimStore[Const::ResultCountPerThread];
					TDisp pxDisp[Const::ResultCountPerThread];

					TDispMinMax uDispMinMax = make_ushort2(TDisp(EDisparityId::Last), TDisp(EDisparityId::First));

#					pragma unroll
					for (TIdx i = 0; i < Const::ResultCountPerThread; ++i)
					{
						//pxMinRange[i] = make_ushort4(0xFFFF, 0, 0, 0);
						pxSimStore[i].SetZero();
						pxSimStore[i].iMax = -Clu::NumericLimits<TSimValue>::Max();

						pxDisp[i] = TDisp(EDisparityId::Unknown);

						TIdx iIdxX, iIdxY;
						if (AlgoSum::GetResultXY(iIdxX, iIdxY, i))
						{
							// Ensure that a whole Base Patch is inside the image to write a result
							if (c_surfDisp.IsInside(iIdxX + iBlockX + Const::BasePatchSizeX, iIdxY + iBlockY + Const::BasePatchSizeY))
							{
								TDispTuple pixDisp;
								TDisp uDispValue;

								const TIdx iPosX = (iIdxX + iBlockX + Const::BasePatchSizeX / 2) / iScale;
								const TIdx iPosY = (iIdxY + iBlockY + Const::BasePatchSizeY / 2) / iScale;

								c_surfDispInit.Read2D(pixDisp, iPosX, iPosY);
								if (/*pixDisp.x >= TDisp(EDisparityId::First) &&*/ pixDisp.x <= TDisp(EDisparityId::Last))
								{
									uDispValue = TDisp(iScale) * pixDisp.x;
									uDispMinMax.x = min(uDispMinMax.x, uDispValue);
									uDispMinMax.y = max(uDispMinMax.y, uDispValue);
									pxDisp[i] = uDispValue;

									//pixDisp.x *= iScale;
								}
								else
								{
									pxDisp[i] = pixDisp.x;
								}


								//c_surfDisp.Write2D(pixDisp
								//	, iIdxX + iBlockX + Const::BasePatchSizeX / 2
								//	, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
							}
						}
					}


					__syncthreads();

					uDispMinMax = MinMaxDisp(uDispMinMax, ShuffleXor(uDispMinMax, 16));
					uDispMinMax = MinMaxDisp(uDispMinMax, ShuffleXor(uDispMinMax, 8));
					uDispMinMax = MinMaxDisp(uDispMinMax, ShuffleXor(uDispMinMax, 4));
					uDispMinMax = MinMaxDisp(uDispMinMax, ShuffleXor(uDispMinMax, 2));
					uDispMinMax = MinMaxDisp(uDispMinMax, ShuffleXor(uDispMinMax, 1));

					//for (int i = 0; i < Const::ResultCountPerThread; ++i)
					//{
					//	int iIdxX, iIdxY;
					//	if (AlgoSum::GetResultXY(iIdxX, iIdxY, i))
					//	{
					//		TDispTuple pixDisp;

					//		//if (uDispMinMax.x > TDisp(EDisparityId::Last) || uDispMinMax.y > TDisp(EDisparityId::Last)
					//		//	|| (uDispMinMax.x == TDisp(EDisparityId::Unknown) && uDispMinMax.y == TDisp(EDisparityId::Unknown)))
					//		if (IsDispValid(uDispMinMax.x) == 0 || IsDispValid(uDispMinMax.y) == 0)
					//		{
					//			uDispMinMax.x = TDisp(EDisparityId::Last) + 1;
					//			uDispMinMax.y = 0;
					//		}

					//		if (uDispMinMax.x > uDispMinMax.y)
					//		{
					//			pixDisp.x = TDisp(EDisparityId::CannotEvaluate);
					//		}
					//		else
					//		{
					//			pixDisp.x = TDisp(EDisparityId::First) + uDispMinMax.y - uDispMinMax.x;
					//		}

					//		c_surfDisp.Write2D(pixDisp
					//			, iIdxX + iBlockX + Const::BasePatchSizeX / 2
					//			, iIdxY + iBlockY + Const::BasePatchSizeY / 2);

					//	}
					//}

					//return;

					if (IsDispValid(uDispMinMax.x) == 0 || IsDispValid(uDispMinMax.y) == 0 || uDispMinMax.x > uDispMinMax.y)
					{
						//uDispMinMax.x = TDisp(EDisparityId::Last) + 1;
						//uDispMinMax.y = 0;

						TDispTuple pixDisp;
						pixDisp.x = TDisp(EDisparityId::Unknown);
						pixDisp.y = 0;
						pixDisp.z = 0;
						pixDisp.w = 0;

						for (int i = 0; i < Const::ResultCountPerThread; ++i)
						{
							pixDisp.x = pxDisp[i];

							int iIdxX, iIdxY;
							if (AlgoSum::GetResultXY(iIdxX, iIdxY, i))
							{
								// Ensure that a whole Base Patch is inside the image to write a result
								if (c_surfDisp.IsInside(iIdxX + iBlockX + Const::BasePatchSizeX, iIdxY + iBlockY + Const::BasePatchSizeY))
								{
									c_surfDisp.Write2D(pixDisp
										, iIdxX + iBlockX + Const::BasePatchSizeX / 2
										, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
								}
							}
						}

					}
					else
					{

						//TDispTuple pixDisp;
						//pixDisp.x = TDisp(EDisparityId::First) + uDispMinMax.y - uDispMinMax.x;
						//pixDisp.y = 0;
						//pixDisp.z = 0;
						//pixDisp.w = 0;

						//for (int i = 0; i < Const::ResultCountPerThread; ++i)
						//{
						//	int iIdxX, iIdxY;
						//	if (AlgoSum::GetResultXY(iIdxX, iIdxY, i))
						//	{
						//		c_surfDisp.Write2D(pixDisp
						//			, iIdxX + iBlockX + Const::BasePatchSizeX / 2
						//			, iIdxY + iBlockY + Const::BasePatchSizeY / 2);
						//	}
						//}
						//return;

						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						// Read an image area into the cache
						//xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);

						if (c_surfImgL.IsRectInside(iBlockX, iBlockY, Const::TestBlockSizeX, Const::TestBlockSizeY))
						{
							xCacheBasePatch.ReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
						}
						else
						{
							xCacheBasePatch.SafeReadFromSurf<Const::TestBlockSizeX, Const::TestBlockSizeY>(c_surfImgL, iBlockX, iBlockY);
						}
						__syncthreads();


						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
						// Start disparity evaluation
						//static const int c_iInitialDispInc = 1;
						const TIdx iDispStart = TIdx(uDispMinMax.x < 4 ? 0 : uDispMinMax.x - 4);
						const TIdx iDispEnd = TIdx(uDispMinMax.y + 4);

						//Debug::Run([&]()
						//{
						//	unsigned short uDelta = uDispMinMax.y - uDispMinMax.x;

						//	if (Debug::IsThread(0, 0) && uDelta > 50)
						//	{
						//		printf("(%d, %d) >> %d, %d: %d, %d\n", blockIdx.x, blockIdx.y, uDispMinMax.x, uDispMinMax.y, iDispStart, iDispEnd);
						//	}
						//});

						for (TIdx iDispIdx = iDispStart; iDispIdx <= iDispEnd; iDispIdx += Const::SubDispCount)
						{
							//const TIdx iTestX = iBlockX - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange / 2 + iDispIdx;

							//const TIdx iTestX = iBlockX + (1 - 2 * c_xPars._iIsLeftToRight) * (iDispIdx - c_xPars._iRightToLeftOffsetX - c_xPars.iDispRange / 2)
							//				- c_xPars._iIsLeftToRight * (Const::SubDispCount - 1);

							const TIdx iTestX = c_xPars.xDispConfig.MapPixelPos(iBlockX, iDispIdx, c_xPars._iIsLeftToRight)
								- c_xPars._iIsLeftToRight * (Const::SubDispCount - 1);


							const TIdx iTestY = iBlockY;

							if (c_surfImgR.Format().IsRectInside(iTestX, iTestY, Const::TestBlockSizeX + Const::SubDispCount - 1, Const::TestBlockSizeY))
							{
								xCacheTestPatch.ReadFromSurf<Const::TestBlockSizeX + Const::SubDispCount, Const::TestBlockSizeY>(c_surfImgR, iTestX, iTestY);
								__syncthreads();

								const TIdx iRelDispCount = min(Const::SubDispCount, iDispEnd - iDispStart + 1);
								for (TIdx iRelDisp = 0; iRelDisp < iRelDispCount; ++iRelDisp)
								{
									EvalSim<Const, TSimType, TPixelSrc>(pxSimStore, pxDisp
										, xCacheBasePatch, xCacheSumY, xCacheTestPatch
										, TIdx(c_xPars._iIsLeftToRight * (Const::SubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
										, TDisp(iDispIdx + iRelDisp), TDisp(1));
								}
							}
							// If not a whole SubDispCount block fits into the image, try to use a smaller block
							else
							{
								const _SImageFormat &xF = c_surfImgR.Format();

								if (iTestY >= 0 && iTestY + Const::TestBlockSizeY <= xF.iHeight)
								{
									int iDeltaX, iStartX;

									if (iTestX >= 0)
									{
										iDeltaX = xF.iWidth - iTestX - Const::TestBlockSizeX;
										iStartX = iTestX;

										//if (threadIdx.x == 0)
										//{
										//	printf("%d, %d > (%d) : %d - %d - %d = %d -> Start: %d\n"
										//		, blockIdx.x, blockIdx.y, iDispIdx, xF.iWidth, iTestX, Const::TestBlockSizeX, iDeltaX, iStartX);
										//}
									}
									else
									{
										iDeltaX = iTestX + (Const::SubDispCount - 1);
										iStartX = 0;

										//if (threadIdx.x == 0)
										//{
										//	printf("%d, %d > (%d) : %d + (%d - 1) = %d -> Start: %d\n"
										//		, blockIdx.x, blockIdx.y, iDispIdx, iTestX, Const::SubDispCount - 1, iDeltaX, iStartX);
										//}
									}

									if (iDeltaX >= 0)
									{
										const int iSubDispCount = iDeltaX + 1;

										xCacheTestPatch.ReadFromSurfVarWidth<Const::TestBlockSizeY>(c_surfImgR, iStartX, iTestY, Const::TestBlockSizeX + iSubDispCount - 1);
										__syncthreads();

										for (TIdx iRelDisp = 0; iRelDisp < iSubDispCount; ++iRelDisp)
										{
											EvalSim<Const, TSimType, TPixelSrc>(pxSimStore, pxDisp
												, xCacheBasePatch, xCacheSumY, xCacheTestPatch
												, TIdx(c_xPars._iIsLeftToRight * (iSubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
												, TDisp(iDispIdx + iRelDisp), TDisp(1));
										}

									}
								}
							}
						}

						__syncthreads();

						WriteDisparity<t_eConfig, TDispTuple, TDisp, TComponent>(pxSimStore, pxDisp, iBlockX, iBlockY);
					}
					__syncthreads();

				}


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
				using Const = Kernel::Constants<Config::PatchCountY_Pow2, Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

				m_xPars._iBlockOffsetX = 0;
				m_xPars._iBlockOffsetY = 0;
				m_xPars._iIsLeftToRight = (m_xPars.bLeftToRight ? 1 : 0);

				EvalThreadConfig_ReadDepBlockSize(xDevice, xFormat
					, Const::BasePatchCountX, Const::BasePatchCountY
					, Const::TestBlockSizeX, Const::TestBlockSizeY
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
				case EConfig::Patch_5x5:
					_DoConfigure<EConfig::Patch_5x5>(xDevice, xFormat);
					break;

				case EConfig::Patch_9x9:
					_DoConfigure<EConfig::Patch_9x9>(xDevice, xFormat);
					break;

				case EConfig::Patch_11x11:
					_DoConfigure<EConfig::Patch_11x11>(xDevice, xFormat);
					break;

				case EConfig::Patch_15x15:
					_DoConfigure<EConfig::Patch_15x15>(xDevice, xFormat);
					break;

				default:
					throw CLU_EXCEPTION("Invalid algorithm configuration.");
				}

			}


			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Select use disp input. </summary>
			///
			/// <typeparam name="TPixelDisp">  	Type of the pixel disp. </typeparam>
			/// <typeparam name="TPixelSrc">   	Type of the pixel source. </typeparam>
			/// <typeparam name="t_ePatchSize">	Type of the patch size. </typeparam>
			/// <typeparam name="t_eSimType">  	Type of the simulation type. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<typename TPixelDisp, typename TPixelSrc, EConfig t_ePatchSize, typename TSimType>
			void CDriver::_SelectUseDispInput()
			{
				if (m_xPars.iUseDispInput)
				{
					Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, t_ePatchSize, TSimType>
						CLU_KERNEL_CONFIG()();
				}
				else
				{
					Kernel::Matcher<TPixelDisp, TPixelSrc, t_ePatchSize, TSimType>
						CLU_KERNEL_CONFIG()();
				}
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Select simulation type. </summary>
			///
			/// <typeparam name="TPixelDisp">  	Type of the pixel disp. </typeparam>
			/// <typeparam name="TPixelSrc">   	Type of the pixel source. </typeparam>
			/// <typeparam name="t_ePatchSize">	Type of the patch size. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<typename TPixelDisp, typename TPixelSrc, EConfig t_ePatchSize>
			void CDriver::_SelectSimType()
			{
				switch (m_xPars.eSimType)
				{
				case ESimType::AbsoluteDifference:
					_SelectUseDispInput<TPixelDisp, TPixelSrc, t_ePatchSize, SSimType_AbsoluteDifference>();
					break;

				case ESimType::SignedProduct:
					_SelectUseDispInput<TPixelDisp, TPixelSrc, t_ePatchSize, SSimType_SignedProduct>();
					break;

				default:
					throw CLU_EXCEPTION("Given similarity type not supported.");
				}
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Executes the process operation. </summary>
			///
			/// <typeparam name="TPixelDisp">	Type of the pixel disp. </typeparam>
			/// <typeparam name="TPixelSrc"> 	Type of the pixel source. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<typename TPixelDisp, typename TPixelSrc>
			void CDriver::_SelectPatchSize()
			{
				switch (m_xPars.eConfig)
				{
				case EConfig::Patch_5x5:
					_SelectSimType<TPixelDisp, TPixelSrc, EConfig::Patch_5x5>();
					break;

				case EConfig::Patch_9x9:
					_SelectSimType<TPixelDisp, TPixelSrc, EConfig::Patch_9x9>();
					break;

				case EConfig::Patch_11x11:
					_SelectSimType<TPixelDisp, TPixelSrc, EConfig::Patch_11x11>();
					break;

				case EConfig::Patch_15x15:
					_SelectSimType<TPixelDisp, TPixelSrc, EConfig::Patch_15x15>();
					break;

				default:
					throw CLU_EXCEPTION("Given patch size not supported.");
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
				, const Clu::Cuda::_CDeviceSurface& xImageDispInit
				, const Clu::Cuda::_CDeviceSurface& xImageDebug)
			{
				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDisp, &xImageDisp, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDispInit, &xImageDispInit, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfImgL, &xImageL, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfImgR, &xImageR, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				Clu::Cuda::MemCpyToSymbol(Kernel::c_surfDebug, &xImageDebug, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

				if (!xImageDisp.IsOfType<TPixelDisp>())
				{
					throw CLU_EXCEPTION("Given disparity image has incorrect type");
				}

				if (xImageL.IsOfType<Clu::TPixel_Lum_UInt8>()
					&& xImageR.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					_SelectPatchSize<TPixelDisp, Clu::TPixel_Lum_UInt8>();
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}
			}
		} // namespace BlockMatcherAW_NCC
	} // namespace Cuda

} // namespace Clu
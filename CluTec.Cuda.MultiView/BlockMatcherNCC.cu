////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      BlockMatcherNCC.cu
//
// summary:   block matcher ncc class
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

#include "BlockMatcherNCC.h"

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
#include "CluTec.Cuda.ImgProc/Kernel.Algo.ArraySum_W32H16.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace BlockMatcherNCC
		{

			namespace Kernel
			{

				using namespace Clu::Cuda::Kernel;

				using TIdx = short;

				template<int t_iPatchSizeX, int t_iPatchSizeY
					, int t_iWarpsPerBlockX, int t_iWarpsPerBlockY>
					struct Constants
				{
					// Warps per block
					static const TIdx WarpsPerBlockX = t_iWarpsPerBlockX;
					static const TIdx WarpsPerBlockY = t_iWarpsPerBlockY;

					// Thread per warp
					static const TIdx ThreadsPerWarp = 32;
					static const TIdx ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;

					using AlgoSum = Clu::Cuda::Kernel::AlgoArraySum_W32H16<ThreadsPerBlockX, t_iPatchSizeX, t_iPatchSizeY>;



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
					using TData = T;

					__device__ __forceinline__ void SetZero()
					{
						uPrev2 = uPrev = uMax = uNext = uNext2 = uLast = T(0);
					}

					T uPrev2;
					T uPrev;
					T uMax;
					T uNext;
					T uNext2;
					T uLast;
				};


				__constant__ _SParameter c_xPars;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDisp;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDispInit;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgL;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfImgR;
				__constant__ Clu::Cuda::_CDeviceSurface c_surfDebug;

				template<typename TResult, uint32_t t_uShift, typename TValue>
				__device__ __forceinline__ TResult CrossCorrelation(TValue xValA, TValue xValB)
				{
					return (TResult)((int32_t(xValA) * int32_t(xValB)) >> t_uShift);
				}

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

				template<typename Const, typename TPixelSurf, typename TSimVal, typename TSimStore, typename TDisp, typename TCacheBasePatch
					, typename TCacheSum, typename TCacheTestPatch, typename TCacheMsd>
					__device__ void EvalSim(TSimStore* pxSimStore
						, TDisp* pxDisp
						, TCacheBasePatch& xCacheBasePatch
						, TCacheSum& xCacheSumY1, TCacheSum& xCacheSumY2, TCacheSum& xCacheSumY3
						, TCacheTestPatch& xCacheTestPatch
						, TCacheMsd& xCacheBaseMean
						, TCacheMsd& xCacheBaseStdDev
						//, Clu::Cuda::_CDeviceSurface& surfImgR
						, TIdx iBlockX, TIdx iBlockY, TDisp iDispIdx, const TDisp uDispIdxInc, float fSadThresh)
				{
					using TSum = typename TCacheSum::TElement;
					const float fBasePatchElementCount = float(Const::BasePatchElementCount);


					Const::AlgoSum::SumCache3(xCacheSumY1, xCacheSumY2, xCacheSumY3,
						[&xCacheBasePatch, &xCacheTestPatch, iBlockX, iBlockY](TSum &iVal1, TSum &iVal2, TSum &iVal3, TIdx iIdxX, TIdx iIdxY)
					{
						auto pixValBase = xCacheBasePatch.At(iIdxX, iIdxY);
						auto pixValTest = xCacheTestPatch.At(iBlockX + iIdxX, iBlockY + iIdxY);

						iVal1 = TSum(pixValTest.x);
						iVal2 = iVal1 * iVal1;
						iVal3 = TSum(pixValBase.x) * iVal1;
					},
						[&fBasePatchElementCount
						, &xCacheBaseMean, &xCacheBaseStdDev
						, &pxSimStore, &pxDisp, &iDispIdx, &uDispIdxInc, &fSadThresh]
						(TSum& iValue1, TSum& iValue2, TSum& iValue3, TIdx iResultIdx, TIdx iIdxX, TIdx iIdxY)
					{
						if (pxDisp[iResultIdx] > TDisp(EDisparityId::Last))
						{
							return;
						}

						auto xSimStore = pxSimStore[iResultIdx];

						// Get cross product sum.
						float fCrossMean = (float(iValue3) / fBasePatchElementCount); // / 255.0f; //(255.0f * 255.0f);

						// Extract Mean
						const float fBaseMean = /*__half2float*/(xCacheBaseMean.At(iIdxX + iResultIdx, iIdxY));// / 255.0f;
						// Extract Std.Dev. assuming that value had been multiplied by 4 before it was stored.
						const float fBaseStdDev = /*__half2float*/(xCacheBaseStdDev.At(iIdxX + iResultIdx, iIdxY));// / 255.0f;

						// Extract Mean
						const float fTestMean = float(iValue1) / fBasePatchElementCount;// / 255.0f;
						// Extract Std.Dev. assuming that value had been multiplied by 4 before it was stored.
						const float fTestMean2 = float(iValue2) / fBasePatchElementCount;// / 255.0f;

						// Evaluate test standard deviation
						const float fTestStdDev = sqrtf(fTestMean2 - fTestMean * fTestMean);

						// Evaluate NCC
						float fNorm = fBaseStdDev * fTestStdDev;

						float fNcc;

						//fNcc = (fCrossMean - fBaseMean * fTestMean) / fNorm;

						//if (Debug::IsBlock(3, 12) && iIdxX + iResultIdx == 6 && iIdxY == 3)
						//{
						//	printf("Cross: %g, Mean B/T: %g, %g; SD B/T: %g, %g; Norm: %g; NCC: %g\n"
						//		, fCrossMean, fBaseMean, fTestMean, fBaseStdDev, fTestStdDev, fNorm, fNcc);
						//}

						//if (fNcc > 0.9f)
						//{
						//	printf("%d, %d, %d, %d: %g, %g\n", blockIdx.x, blockIdx.y, iIdxX + iResultIdx, iIdxY, fNorm, fNcc);
						//}


						// If the product of the standard deviations is too small, then do not evaluate NCC
						// but simply set it to zero.
						if (fNorm < 1.0f)
						{
							fNcc = 0.0f;
						}
						else
						{
							fNcc = (fCrossMean - fBaseMean * fTestMean) / fNorm;
						}

						float fOrigNcc = fNcc;

						// Stretch the top 0.2 units of the NCC range of [-1, 1] to 16bit.
						//fNcc = (fNcc - 0.8f) / 0.2f;
						fNcc = max(0.0f, fNcc);

						const float fScale = (float)Clu::NumericLimits<TSimVal>::Max();
						TSimVal uNcc = TSimVal(min(fScale, fNcc * fScale));

						//if (Debug::IsBlock(26, 29) && iIdxX + iResultIdx == 2 && iIdxY == 9)
						////if (fOrigNcc > 0.95f)
						//{
						//	printf("%d: fOrigNcc: %g, fNcc: %g, uNcc: %d\n"
						//		, iDispIdx
						//		, fOrigNcc, fNcc, uNcc);

						//	//printf("%d, %d, %d, %d: fOrigNcc: %g, fNcc: %g, uNcc: %d\n"
						//	//	, blockIdx.x, blockIdx.y, iIdxX + iResultIdx, iIdxY
						//	//	, fOrigNcc, fNcc, uNcc);
						//}


						//if (blockIdx.x == 12 && blockIdx.y == 10 && threadIdx.x == 26 && iResultIdx == 0)
						//{
						//	printf("Disp Idx: %d, Opt: %d | SAD [%d -> %d -> %d -> %d -> %d], prev: %d, cur: %d\n"
						//		, iDispIdx, pxDisp[iResultIdx]
						//		, xMinRange.uPrev2, xMinRange.uPrev, xMinRange.uMax, xMinRange.uNext, xMinRange.uNext2, xMinRange.uLast, iValue);
						//}

						//float fMin = float(iValue) / float(Const::BasePatchElementCount);

						// xMinRange.x: Minimum value
						// xMinRange.w: Value at previous disparity step
						// xMinRange.y: Value at disparity step just before minimum
						// xMinRange.z: Value at disparity step just after minimum

						const TSimVal uIsAboveMax = TSimVal(uNcc > xSimStore.uMax);
						const TSimVal uIsJustBehindMin = (1 - uIsAboveMax)
							* TSimVal(pxDisp[iResultIdx] + uDispIdxInc == (iDispIdx + TDisp(EDisparityId::First)));
						const TSimVal uIsJust2BehindMin = (1 - uIsAboveMax)
							* TSimVal(pxDisp[iResultIdx] + 2 * uDispIdxInc == (iDispIdx + TDisp(EDisparityId::First)));

						xSimStore.uPrev2 = (1 - uIsAboveMax) * xSimStore.uPrev2 + uIsAboveMax * xSimStore.uPrev;

						// We have a new minimum, so set the value before minimum to previous value.
						xSimStore.uPrev = (1 - uIsAboveMax) * xSimStore.uPrev + uIsAboveMax * xSimStore.uLast;

						// Set the minimum
						xSimStore.uMax = (1 - uIsAboveMax) * xSimStore.uMax + uIsAboveMax * uNcc;

						// Set the value after the minimum also to the minimum.
						// Can only be set properly in the next step.
						xSimStore.uNext = (1 - uIsAboveMax) * xSimStore.uNext + uIsAboveMax * uNcc;
						xSimStore.uNext2 = (1 - uIsAboveMax) * xSimStore.uNext2 + uIsAboveMax * uNcc;

						// Set disparity at minimum
						pxDisp[iResultIdx] = (1 - uIsAboveMax) * pxDisp[iResultIdx] + uIsAboveMax * (iDispIdx + TDisp(EDisparityId::First));

						// The current value is not a new minimum.
						// If the previous value is the minimum then set the value after the minimum to this value.
						xSimStore.uNext = (1 - uIsJustBehindMin) * xSimStore.uNext + uIsJustBehindMin * uNcc;
						xSimStore.uNext2 = (1 - uIsJust2BehindMin) * xSimStore.uNext2 + uIsJust2BehindMin * uNcc;


						// Set current value to previous value
						xSimStore.uLast = uNcc;

						pxSimStore[iResultIdx] = xSimStore;

					});


				}

				template<typename Const, typename TCacheMsd, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalMsd(TCacheMsd& xCacheMean, TCacheMsd& xCacheStdDev
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY)
				{
					using TSum = typename TCacheSum::TElement;

					const float fBasePatchElementCount = float(Const::BasePatchElementCount);
					TSum piMean[Const::ResultCountPerThread];

					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch](TIdx iIdxX, TIdx iIdxY)
					{
						auto pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						return TSum(pixVal1.x);
					},
						[&piMean](TSum& iValue, TIdx iRelX, TIdx iIdxX, TIdx iIdxY)
					{
						piMean[iRelX] = iValue;
					});

					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch](TIdx iIdxX, TIdx iIdxY)
					{
						auto pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						return TSum(pixVal1.x) * TSum(pixVal1.x);
					},
						[&piMean, &fBasePatchElementCount, &xCacheMean, &xCacheStdDev](TSum& iValue, TIdx iRelX, TIdx iIdxX, TIdx iIdxY)
					{
						float fMean = float(piMean[iRelX]) / fBasePatchElementCount;
						float fMean2 = float(iValue) / fBasePatchElementCount;

						xCacheMean.At(iIdxX + iRelX, iIdxY) = /*__float2half*/(fMean);
						xCacheStdDev.At(iIdxX + iRelX, iIdxY) = /*__float2half*/(sqrtf(fMean2 - fMean * fMean));
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

				template<typename Const, typename TDisp, typename TCacheBasePatch, typename TCacheSum>
				__device__ void EvalGrad(TDisp* pxDisp
					, TCacheBasePatch& xCacheBasePatch
					, TCacheSum& xCacheSumY
					, float fGradThresh)
				{
					using TSum = typename TCacheSum::TElement;

					Const::AlgoSum::SumCache(xCacheSumY,
						[&xCacheBasePatch](TIdx iIdxX, TIdx iIdxY)
					{
						auto& pixVal1 = xCacheBasePatch.At(iIdxX, iIdxY);
						auto& pixVal2 = xCacheBasePatch.At(TIdx(iIdxX + 1), TIdx(iIdxY));
						return AbsoluteDifference<TSum>(pixVal1.x, pixVal2.x);
						//return CensusDifference<TSum>(pixVal1.x, pixVal2.x);
					},
						[&pxDisp, &fGradThresh](TSum& iValue, TIdx iRelX, TIdx iIdxX, TIdx iIdxY)
					{
						float fValue = float(iValue) / float(Const::BasePatchElementCount);

						if (pxDisp[iRelX] == TDisp(EDisparityId::Unknown) && fValue < fGradThresh)
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


				template<EConfig t_eConfig, typename TDispTuple, typename TDisp, typename TSimStore>
				__device__ __forceinline__ void WriteDisparity(TSimStore* pxSimStore, TDisp* pxDisp, TIdx iBlockX, TIdx iBlockY)
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;
					using AlgoSum = typename Const::AlgoSum;
					using TSimVal = typename TSimStore::TData;

					//if (!Debug::IsBlock(13, 15) && !Debug::IsBlock(14, 15))
					//{
					//	return;
					//}

					for (TIdx iResultIdx = 0; iResultIdx < Const::ResultCountPerThread; ++iResultIdx)
					{
						TIdx iIdxX, iIdxY;
						if (AlgoSum::GetResultXY(iIdxX, iIdxY, iResultIdx))
						{
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

									float fNccMax = /*0.8f + 0.2f * */(float(xSimStore.uMax) / float(Clu::NumericLimits<TSimVal>::Max()));

									TIdx iIsAboveMax = TIdx(fNccMax >= c_xPars.fNccThresh);

									// ///////////////////////////////////////////////////////////////////////////////////////////////////
									// TODO: Scale these difference in the same way as the NCC value above, and adjust the threshold types.
									// ///////////////////////////////////////////////////////////////////////////////////////////////////
									TIdx iDiff1 = max(abs(TIdx(xSimStore.uPrev) - TIdx(xSimStore.uMax)), abs(TIdx(xSimStore.uPrev2) - TIdx(xSimStore.uMax)));
									TIdx iDiff2 = max(abs(TIdx(xSimStore.uNext) - TIdx(xSimStore.uMax)), abs(TIdx(xSimStore.uNext2) - TIdx(xSimStore.uMax)));;

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
									//		, xMinRange.uPrev2, xMinRange.uPrev, xMinRange.uMax, xMinRange.uNext, xMinRange.uNext2, xMinRange.uLast
									//		, iDiff1, iDiff2, c_xPars.iMinDeltaThresh);
									//}

									//pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(xMinRange.x, xMinRange.x, iDiff1, iDiff2);


									if (iIsAboveMax)
									{
										TIdx iIsDispSaturated = (pxDisp[iResultIdx] <= TDisp(EDisparityId::First) + TDisp(c_xPars.xDispConfig.Min()) + 1
											|| pxDisp[iResultIdx] >= TDisp(EDisparityId::First) + c_xPars.xDispConfig.Max() - 1);

										if (iIsDispSaturated)
										{
											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::Saturated), xSimStore.uMax, iDiff1, iDiff2);
										}
										else if (iDiff1 >= c_xPars.iMinDeltaThresh && iDiff2 >= c_xPars.iMinDeltaThresh)
										{
											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(pxDisp[iResultIdx], xSimStore.uMax, iDiff1, iDiff2);
										}
										else
										{
											//if (blockIdx.x == 12 && blockIdx.y == 10)
											//{
											//	printf("Not Specific: %d [%d]\n", threadIdx.x, iResultIdx);
											//}

											pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotSpecific), xSimStore.uMax, iDiff1, iDiff2);
										}
									}
									else
									{
										pixValue = Clu::Cuda::Make<TDispTuple, TDisp>(TDisp(EDisparityId::NotFound), xSimStore.uMax, iDiff1, iDiff2);
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

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig>
				__global__ void Matcher()
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoSum = typename Const::AlgoSum;

					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;

					using TComponent = typename TPixelSrc::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TSum = int; //short;// unsigned short;
					using TSimVal = unsigned char;
					using TSimStore = SSimStore<TSimVal>;
					using TFloat = float;

					using TCachePatch = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheMsd = Clu::Cuda::Kernel::CArrayCache<float
						, Const::BasePatchCountX, Const::BasePatchCountY
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
					__shared__ TCacheMsd xCacheBaseMean;
					__shared__ TCacheMsd xCacheBaseStdDev;
					__shared__ TCacheSum xCacheSumY1;
					__shared__ TCacheSum xCacheSumY2;
					__shared__ TCacheSum xCacheSumY3;



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
						pxSimStore[i].uMax = 0;
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
					EvalGrad<Const>(pxDisp, xCacheBasePatch, xCacheSumY1, c_xPars.fGradThresh);

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
					// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Evaluate Mean & Std.Dev. for base patch
					EvalMsd<Const>(xCacheBaseMean, xCacheBaseStdDev, xCacheBasePatch, xCacheSumY1);

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
								EvalSim<Const, TPixelSrc, TSimVal>(pxSimStore, pxDisp
									, xCacheBasePatch, xCacheSumY1, xCacheSumY2, xCacheSumY3, xCacheTestPatch
									, xCacheBaseMean, xCacheBaseStdDev
									, TIdx(c_xPars._iIsLeftToRight * (Const::SubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
									, TDisp(iDispIdx + iRelDisp), TDisp(1), c_xPars.fNccThresh);
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
										EvalSim<Const, TPixelSrc, TSimVal>(pxSimStore, pxDisp
											, xCacheBasePatch, xCacheSumY1, xCacheSumY2, xCacheSumY3, xCacheTestPatch
											, xCacheBaseMean, xCacheBaseStdDev
											, TIdx(c_xPars._iIsLeftToRight * (iSubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
											, TDisp(iDispIdx + iRelDisp), TDisp(1), c_xPars.fNccThresh);
									}

								}
							}
						}
					}

					__syncthreads();

					WriteDisparity<t_eConfig, TDispTuple, TDisp>(pxSimStore, pxDisp, iBlockX, iBlockY);

					__syncthreads();

				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Matcher use disparity. </summary>
				///
				/// <typeparam name="TPixelDisp">	Type of the pixel disp. </typeparam>
				/// <typeparam name="TPixelSrc"> 	Type of the pixel source. </typeparam>
				/// <typeparam name="t_eConfig"> 	Type of the configuration. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelDisp, typename TPixelSrc, EConfig t_eConfig>
				__global__ void MatcherUseDisparity()
				{
					using Config = SConfig<t_eConfig>;
					using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

					using AlgoSum = typename Const::AlgoSum;

					using TElement = typename Clu::Cuda::SPixelTypeInfo<TPixelSrc>::TElement;

					using TComponent = typename TPixelSrc::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TSum = int;
					using TSimVal = unsigned char;
					using TSimStore = SSimStore<TSimVal>;
					using TIdx = short;
					using TDispMinMax = ushort2;
					using TFloat = float;

					using TCachePatch = Clu::Cuda::Kernel::CArrayCache<TElement
						, Const::TestBlockSizeX, Const::TestBlockSizeY
						, Const::WarpsPerBlockX, Const::WarpsPerBlockY, 8, 1>;

					using TCacheMsd = Clu::Cuda::Kernel::CArrayCache<TFloat
						, Const::BasePatchCountX, Const::BasePatchCountY
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
					__shared__ TCacheMsd xCacheBaseMean;
					__shared__ TCacheMsd xCacheBaseStdDev;
					__shared__ TCacheSum xCacheSumY1;
					__shared__ TCacheSum xCacheSumY2;
					__shared__ TCacheSum xCacheSumY3;



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
						pxSimStore[i].uMax = 0;

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
						// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						// Evaluate Mean & Std.Dev. for base patch
						EvalMsd<Const>(xCacheBaseMean, xCacheBaseStdDev, xCacheBasePatch, xCacheSumY1);

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
									EvalSim<Const, TPixelSrc, TSimVal>(pxSimStore, pxDisp
										, xCacheBasePatch, xCacheSumY1, xCacheSumY2, xCacheSumY3, xCacheTestPatch
										, xCacheBaseMean, xCacheBaseStdDev
										, TIdx(c_xPars._iIsLeftToRight * (Const::SubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
										, TDisp(iDispIdx + iRelDisp), TDisp(1), c_xPars.fNccThresh);
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
											EvalSim<Const, TPixelSrc, TSimVal>(pxSimStore, pxDisp
												, xCacheBasePatch, xCacheSumY1, xCacheSumY2, xCacheSumY3, xCacheTestPatch
												, xCacheBaseMean, xCacheBaseStdDev
												, TIdx(c_xPars._iIsLeftToRight * (iSubDispCount - 1) + (1 - 2 * c_xPars._iIsLeftToRight) * iRelDisp), TIdx(0)
												, TDisp(iDispIdx + iRelDisp), TDisp(1), c_xPars.fNccThresh);
										}

									}
								}
							}
						}

						__syncthreads();

						WriteDisparity<t_eConfig, TDispTuple, TDisp>(pxSimStore, pxDisp, iBlockX, iBlockY);
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
				using Const = Kernel::Constants<Config::PatchSizeX, Config::PatchSizeY, Config::WarpsPerBlockX, Config::WarpsPerBlockY>;

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
				case EConfig::Patch_11x11:
					_DoConfigure<EConfig::Patch_11x11>(xDevice, xFormat);
					break;

				case EConfig::Patch_9x9:
					_DoConfigure<EConfig::Patch_9x9>(xDevice, xFormat);
					break;

				case EConfig::Patch_5x5:
					_DoConfigure<EConfig::Patch_5x5>(xDevice, xFormat);
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
					switch (m_xPars.eConfig)
					{
					case EConfig::Patch_11x11:
						Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, EConfig::Patch_11x11>
							CLU_KERNEL_CONFIG()();
						break;

					case EConfig::Patch_9x9:
						Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, EConfig::Patch_9x9>
							CLU_KERNEL_CONFIG()();
						break;

					case EConfig::Patch_5x5:
						Kernel::MatcherUseDisparity<TPixelDisp, TPixelSrc, EConfig::Patch_5x5>
							CLU_KERNEL_CONFIG()();
						break;

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}
				}
				else
				{
					switch (m_xPars.eConfig)
					{
					case EConfig::Patch_11x11:
						Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_11x11>
							CLU_KERNEL_CONFIG()();
						break;

					case EConfig::Patch_9x9:
						Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_9x9>
							CLU_KERNEL_CONFIG()();
						break;

					case EConfig::Patch_5x5:
						Kernel::Matcher<TPixelDisp, TPixelSrc, EConfig::Patch_5x5>
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
					_DoProcess<TPixelDisp, Clu::TPixel_Lum_UInt8>();
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}
			}
		} // namespace BlockMatcherAW_NCC
	} // namespace Cuda

} // namespace Clu
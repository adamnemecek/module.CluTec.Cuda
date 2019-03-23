////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Kernel.Algo.AdaptWnd_W32H16.h
//
// summary:   Declares the kernel. algo. adapt window 32 h 16 class
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

#pragma once

#include "CluTec.Cuda.Base/Kernel.ArrayCache_W16.h"
#include "Kernel.Algo.ArraySum_W32H16.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Kernel
		{
			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Adaptive window optimized for a sum array size of 32 columns by 16 rows. </summary>
			///
			/// <typeparam name="t_iThreadsPerBlockX">	Type of the threads per block x coordinate. </typeparam>
			/// <typeparam name="t_iSubPatchSizeX">   	Type of the sub patch size x coordinate. </typeparam>
			/// <typeparam name="t_iSubPatchSizeY">   	Type of the sub patch size y coordinate. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<int t_iThreadsPerBlockX, int t_iSubPatchSizeX, int t_iSubPatchSizeY>
			struct AlgoAdaptWnd_W32H16
			{
				enum class EType
				{
					Two_OutOf_Eight = 0,
					Two_OutOf_Four,
					Three_OutOf_Eight,
				};

				static const int ThreadsPerBlockX = t_iThreadsPerBlockX;
				static const int SubPatchCountY = 16;

				static const int SubPatchSizeX = t_iSubPatchSizeX;
				static const int SubPatchSizeY = t_iSubPatchSizeY;

				static const int MinSubPatchCountY = 2 * SubPatchSizeY + 1;

				static_assert(MinSubPatchCountY <= SubPatchCountY, "The number of sub-patches in y-direction is too small");


				using AlgoSum = AlgoArraySum_W32H16<ThreadsPerBlockX, SubPatchSizeX, SubPatchSizeY>;

				static const int SubPatchCountX = AlgoSum::PatchCountX;

				static const int PatchSizeX = 3 * SubPatchSizeX;
				static const int PatchSizeY = 3 * SubPatchSizeY;

				static const int SubPatchElementCount = SubPatchSizeX * SubPatchSizeY;
				static const int EffectivePatchElementCount = 3 * SubPatchElementCount;

				static const int PatchCountX = SubPatchCountX - PatchSizeX + SubPatchSizeX;
				static_assert( PatchCountX > 0, "The sub-patch width is too large." );

				static const int PatchCountY = SubPatchCountY - PatchSizeY + SubPatchSizeY;
				static_assert( PatchCountY > 0, "The sub-patch height is too large." );

				static const int DataArraySizeX = PatchCountX + PatchSizeX - 1;
				static_assert(DataArraySizeX == AlgoSum::DataArraySizeX, "Programming error");

				static const int DataArraySizeY = PatchCountY + PatchSizeY - 1;
				static_assert(DataArraySizeY == AlgoSum::DataArraySizeY, "Programming error");

				static const int SumCacheSizeX = AlgoSum::SumCacheSizeX;
				static const int SumCacheSizeY = AlgoSum::SumCacheSizeY;

				static const int TotalPatchCount = PatchCountX * PatchCountY;
				static const int EvalLoopCount = TotalPatchCount / ThreadsPerBlockX + int(TotalPatchCount % ThreadsPerBlockX > 0);

#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

				__device__ static void PrintStaticValues()
				{
					PRINT(ThreadsPerBlockX);
					PRINT(SubPatchCountX);
					PRINT(SubPatchCountY);
					PRINT(MinSubPatchCountY);
					PRINT(SubPatchSizeX);
					PRINT(SubPatchSizeY);
					PRINT(PatchSizeX);
					PRINT(PatchSizeY);
					PRINT(PatchCountX);
					PRINT(PatchCountY);
					PRINT(DataArraySizeX);
					PRINT(DataArraySizeY);
					PRINT(SumCacheSizeX);
					PRINT(SumCacheSizeY);
					PRINT(TotalPatchCount);
					PRINT(EvalLoopCount);
					//PRINT();
					printf("\n");

					__syncthreads();
				}
#undef PRINT

				template<typename T>
				__device__ static void OrderedInsertMax(T& iFirst, T& iSecond, const T& iValue)
				{
					iSecond = max(iSecond, min(iFirst, iValue));
					iFirst = max(iFirst, iValue);
				}

				template<typename T>
				__device__ static void OrderedInsertMin(T& iFirst, T& iSecond, const T& iValue)
				{
					iSecond = min(iSecond, max(iFirst, iValue));
					iFirst = min(iFirst, iValue);
				}

				template<typename T>
				__device__ static void OrderedInsertMin(T& iFirst, T& iSecond, T& iThird, const T& iValue)
				{
					iThird = min(iThird, max(iSecond, max(iFirst, iValue)));
					iSecond = min(iSecond, max(iFirst, iValue));
					iFirst = min(iFirst, iValue);
				}

				template<typename TIdx>
				__device__ static TIdx GetResultXY(TIdx& iIdxX, TIdx &iIdxY, TIdx iResultIdx)
				{
					const TIdx iEvalPos = (iResultIdx * ThreadsPerBlockX + threadIdx.x);
					iIdxX = iEvalPos % PatchCountX;
					iIdxY = iEvalPos / PatchCountX;

					return iIdxY < PatchCountY;
				}

				template<typename TIdx>
				__device__ static TIdx HasResult(TIdx iResultIdx)
				{
					return ((iResultIdx * ThreadsPerBlockX + threadIdx.x) / PatchCountX) < PatchCountY;
				}

				template<EType t_eType>
				struct SNeighborhood
				{
					template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
					__device__ static void Eval(TSum* pCache, FuncResult funcResult);
				};

				//template<> struct SNeighborhood<EType::Two_OutOf_Eight>
				//{
				//	template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
				//	__device__ static void Eval(TSum* pCache, FuncResult funcResult)
				//	{
				//		for (int iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
				//		{
				//			const int iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
				//			const int iIdxX = iEvalPos % PatchCountX;
				//			const int iIdxY = iEvalPos / PatchCountX;

				//			if (iIdxY >= PatchCountY)
				//			{
				//				break;
				//			}

				//			// Find the 2 smallest sub-patches in the 8-neighborhood of the central sub-patch.
				//			TSum iMin1 = Clu::NumericLimits<TSum>::Max();
				//			TSum iMin2 = Clu::NumericLimits<TSum>::Max();

				//			iMin1 = pCache[iIdxX * t_iStrideX + iIdxY * t_iStrideY];

				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);

				//			TSum iResult = iMin1 + iMin2 + pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY];
				//			funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
				//		}
				//	}
				//};

				//template<> struct SNeighborhood<EType::Three_OutOf_Eight>
				//{
				//	template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
				//	__device__ static void Eval(TSum* pCache, FuncResult funcResult)
				//	{
				//		for (int iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
				//		{
				//			const int iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
				//			const int iIdxX = iEvalPos % PatchCountX;
				//			const int iIdxY = iEvalPos / PatchCountX;

				//			if (iIdxY >= PatchCountY)
				//			{
				//				break;
				//			}

				//			// Find the 3 smallest sub-patches in the 8-neighborhood of the central sub-patch.
				//			TSum iMin1 = Clu::NumericLimits<TSum>::Max();
				//			TSum iMin2 = Clu::NumericLimits<TSum>::Max();
				//			TSum iMin3 = Clu::NumericLimits<TSum>::Max();

				//			iMin1 = pCache[iIdxX * t_iStrideX + iIdxY * t_iStrideY];

				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
				//			OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);

				//			TSum iResult = iMin1 + iMin2 + iMin3 + pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY];
				//			funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
				//		}
				//	}
				//};

				template<> struct SNeighborhood<EType::Two_OutOf_Four>
				{
					template<typename TCache, typename FuncResult>
					__device__ static void Eval(TCache &xCache, FuncResult funcResult)
					{
						// Assumes TSum is signed.
						using TSum = typename TCache::TElement;
						using TIdx = short;

						for (TIdx iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
						{
							const TIdx iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
							const TIdx iIdxX = iEvalPos % PatchCountX;
							const TIdx iIdxY = iEvalPos / PatchCountX;

							if (iIdxY >= PatchCountY)
							{
								break;
							}

							// Find the 2 sub-patches in the 4-neighborhood of the central sub-patch with the highest
							// similarity values.
							TSum iMax1 = -Clu::NumericLimits<TSum>::Max();
							TSum iMax2 = -Clu::NumericLimits<TSum>::Max();

							iMax1 = xCache.At(iIdxX + 1 * SubPatchSizeX, iIdxY + 0 * SubPatchSizeY);

							//if (blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	const int iX = iIdxX + 1 * SubPatchSizeX;
							//	const int iY = iIdxY + 2 * SubPatchSizeY;
							//	int iPos = xCache.IndexAt(iX, iY);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] (%d, %d) >> (%d) %d\n", threadIdx.x, iX, iY, iPos, iBank);
							//}

							OrderedInsertMax(iMax1, iMax2, xCache.At(iIdxX + 0 * SubPatchSizeX, iIdxY + 1 * SubPatchSizeY));
							OrderedInsertMax(iMax1, iMax2, xCache.At(iIdxX + 1 * SubPatchSizeX, iIdxY + 2 * SubPatchSizeY));
							OrderedInsertMax(iMax1, iMax2, xCache.At(iIdxX + 2 * SubPatchSizeX, iIdxY + 1 * SubPatchSizeY));

							TSum iResult = iMax1 + iMax2 + xCache.At(iIdxX + 1 * SubPatchSizeX, iIdxY + 1 * SubPatchSizeY);

							funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
						}
					}
				};

				template<EType t_eType, typename TCache, typename FuncData, typename FuncResult>
				__device__ static void EvalCache(TCache& xCache, FuncData funcData, FuncResult funcResult)
				{
					// Sum all sub-patches and store results in cache
					AlgoSum::SumCacheStore(xCache, funcData);

					__syncthreads();

					SNeighborhood<t_eType>::Eval(xCache, funcResult);

					__syncthreads();
				}


			};
		}
	}
}

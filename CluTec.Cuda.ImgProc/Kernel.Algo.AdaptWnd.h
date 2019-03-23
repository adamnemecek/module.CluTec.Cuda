////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Kernel.Algo.AdaptWnd.h
//
// summary:   Declares the kernel. algo. adapt Windows Form
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

#include "CluTec.Cuda.Base/Kernel.ArrayCache.h"
#include "Kernel.Algo.ArraySum.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Kernel
		{

			template<int t_iThreadsPerBlockX, int t_iSubPatchCountY_Pow2, int t_iSubPatchSizeX, int t_iSubPatchSizeY>
			struct AlgoAdaptWnd
			{
				enum class EType
				{
					Two_OutOf_Eight = 0,
					Two_OutOf_Four,
					Three_OutOf_Eight,
				};

				static const int ThreadsPerBlockX = t_iThreadsPerBlockX;
				static const int SubPatchCountY = 1 << t_iSubPatchCountY_Pow2;

				static const int SubPatchSizeX = t_iSubPatchSizeX;
				static const int SubPatchSizeY = t_iSubPatchSizeY;

				static const int MinSubPatchCountY = 2 * SubPatchSizeY + 1;

				static_assert(MinSubPatchCountY <= SubPatchCountY, "The number of sub-patches in y-direction is too small");


				using AlgoSum = AlgoArraySum<ThreadsPerBlockX, t_iSubPatchCountY_Pow2, SubPatchSizeX, SubPatchSizeY>;

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
					//PRINT();
					printf("\n");

					__syncthreads();
				}
#undef PRINT

				template<typename T>
				__device__ static void OrderedInsert(T& iFirst, T& iSecond, const T& iValue)
				{
					iSecond = min(iSecond, max(iFirst, iValue));
					iFirst = min(iFirst, iValue);
				}

				template<typename T>
				__device__ static void OrderedInsert(T& iFirst, T& iSecond, T& iThird, const T& iValue)
				{
					iThird = min(iThird, max(iSecond, max(iFirst, iValue)));
					iSecond = min(iSecond, max(iFirst, iValue));
					iFirst = min(iFirst, iValue);
				}

				__device__ static int GetResultXY(int& iIdxX, int &iIdxY, int iResultIdx)
				{
					const int iEvalPos = (iResultIdx * ThreadsPerBlockX + threadIdx.x);
					iIdxX = iEvalPos % PatchCountX;
					iIdxY = iEvalPos / PatchCountX;

					return iIdxY < PatchCountY;
				}

				__device__ static int HasResult(int iResultIdx)
				{
					return ((iResultIdx * ThreadsPerBlockX + threadIdx.x) / PatchCountX) < PatchCountY;
				}

				template<EType t_eType>
				struct SNeighborhood
				{
					template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
					__device__ static void Eval(TSum* pCache, FuncResult funcResult);
				};

				template<> struct SNeighborhood<EType::Two_OutOf_Eight>
				{
					template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
					__device__ static void Eval(TSum* pCache, FuncResult funcResult)
					{
						for (int iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
						{
							const int iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
							const int iIdxX = iEvalPos % PatchCountX;
							const int iIdxY = iEvalPos / PatchCountX;

							if (iIdxY >= PatchCountY)
							{
								break;
							}

							// Find the 2 smallest sub-patches in the 8-neighborhood of the central sub-patch.
							TSum iMin1 = Clu::NumericLimits<TSum>::Max();
							TSum iMin2 = Clu::NumericLimits<TSum>::Max();

							iMin1 = pCache[iIdxX * t_iStrideX + iIdxY * t_iStrideY];

							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);

							TSum iResult = iMin1 + iMin2 + pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY];
							funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
						}
					}
				};

				template<> struct SNeighborhood<EType::Three_OutOf_Eight>
				{
					template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
					__device__ static void Eval(TSum* pCache, FuncResult funcResult)
					{
						for (int iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
						{
							const int iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
							const int iIdxX = iEvalPos % PatchCountX;
							const int iIdxY = iEvalPos / PatchCountX;

							if (iIdxY >= PatchCountY)
							{
								break;
							}

							// Find the 3 smallest sub-patches in the 8-neighborhood of the central sub-patch.
							TSum iMin1 = Clu::NumericLimits<TSum>::Max();
							TSum iMin2 = Clu::NumericLimits<TSum>::Max();
							TSum iMin3 = Clu::NumericLimits<TSum>::Max();

							iMin1 = pCache[iIdxX * t_iStrideX + iIdxY * t_iStrideY];

							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, iMin3, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);

							TSum iResult = iMin1 + iMin2 + iMin3 + pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY];
							funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
						}
					}
				};

				template<> struct SNeighborhood<EType::Two_OutOf_Four>
				{
					template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncResult>
					__device__ static void Eval(TSum* pCache, FuncResult funcResult)
					{
						for (int iEvalIdx = 0; iEvalIdx < EvalLoopCount; ++iEvalIdx)
						{
							const int iEvalPos = (iEvalIdx * ThreadsPerBlockX + threadIdx.x);
							const int iIdxX = iEvalPos % PatchCountX;
							const int iIdxY = iEvalPos / PatchCountX;

							if (iIdxY >= PatchCountY)
							{
								break;
							}

							// Find the 2 smallest sub-patches in the 4-neighborhood of the central sub-patch.
							TSum iMin1 = Clu::NumericLimits<TSum>::Max();
							TSum iMin2 = Clu::NumericLimits<TSum>::Max();

							iMin1 = pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 0 * SubPatchSizeY) * t_iStrideY];

							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 0 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 2 * SubPatchSizeY) * t_iStrideY]);
							OrderedInsert(iMin1, iMin2, pCache[(iIdxX + 2 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY]);

							TSum iResult = iMin1 + iMin2 + pCache[(iIdxX + 1 * SubPatchSizeX) * t_iStrideX + (iIdxY + 1 * SubPatchSizeY) * t_iStrideY];
							funcResult(iResult, iEvalIdx, iIdxX, iIdxY);
						}
					}
				};

				template<EType t_eType, int t_iStrideX, int t_iStrideY, typename TSumPtr, typename FuncData, typename FuncResult>
				__device__ static void Eval(TSumPtr pCache, FuncData funcData, FuncResult funcResult)
				{
					// Sum all sub-patches and store results in cache
					AlgoSum::SumCacheStore<t_iStrideX, t_iStrideY>(pCache, funcData);

					__syncthreads();

					//SNeighborhood<t_eType>::Eval<t_iStrideX, t_iStrideY>(pCache, funcResult);
				}



				template<EType t_eType, typename TCache, typename FuncData, typename FuncResult>
				__device__ static void EvalCache(TCache& xCache, FuncData funcData, FuncResult funcResult)
				{
					Eval<t_eType, TCache::StrideX, TCache::StrideY, TCache::TElementPtr>(xCache.DataPointer(), funcData, funcResult);
				}


			};
		}
	}
}

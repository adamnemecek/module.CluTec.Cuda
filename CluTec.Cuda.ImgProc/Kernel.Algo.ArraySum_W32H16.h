////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Kernel.Algo.ArraySum_W32H16.h
//
// summary:   Declares the kernel. algo. array sum w 32 h 16 class
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

namespace Clu
{
	namespace Cuda
	{
		namespace Kernel
		{
			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// An algo array sum optimized for a sum of 32 columns and 16 rows.  
			/// Ideally it uses a ArrayCache_W16 if the 32 columns fit in 16 words.
			/// </summary>
			///
			/// <typeparam name="t_iThreadsPerBlockX">	Type of the threads per block x coordinate. </typeparam>
			/// <typeparam name="t_iPatchCountY_Pow2">	Type of the i patch count y coordinate pow 2. </typeparam>
			/// <typeparam name="t_iPatchSizeX">	  	Type of the patch size x coordinate. </typeparam>
			/// <typeparam name="t_iPatchSizeY">	  	Type of the patch size y coordinate. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<int t_iThreadsPerBlockX, int t_iPatchSizeX, int t_iPatchSizeY>
			struct AlgoArraySum_W32H16
			{
				using TIdx = short;

				static const int ThreadsPerBlockX = t_iThreadsPerBlockX;
				static const int PatchCountY = 16;

				static const int PatchSizeX = t_iPatchSizeX;
				static const int PatchSizeY = t_iPatchSizeY;
				static const int PatchElementCount = PatchSizeX * PatchSizeY;

				static const int DataArraySizeX = ThreadsPerBlockX;
				static const int DataArraySizeY = PatchCountY + PatchSizeY - 1;
				
				static const int ColGroupsPerBlock = ThreadsPerBlockX / PatchCountY;
				static const int ColGroupSizeX = ((DataArraySizeX - PatchSizeX + 1) / ColGroupsPerBlock);
				
				static const int PatchCountX = ColGroupSizeX * ColGroupsPerBlock;

				static const int SumCacheSizeX = DataArraySizeX;
				static const int SumCacheSizeY = PatchCountY;

#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

				__device__ static void PrintStaticValues()
				{
					PRINT(ThreadsPerBlockX);
					PRINT(PatchCountX);
					PRINT(PatchCountY);
					PRINT(PatchElementCount);
					PRINT(DataArraySizeX);
					PRINT(DataArraySizeY);
					PRINT(SumCacheSizeX);
					PRINT(SumCacheSizeY);
					PRINT(ColGroupSizeX);
					//PRINT();
					printf("\n");

					__syncthreads();
				}
#undef PRINT

				__device__ static TIdx RowSumBaseIdxX()
				{
					return (threadIdx.x / PatchCountY) * ColGroupSizeX;
				}

				__device__ static TIdx RowSumBaseIdxY()
				{
					return (threadIdx.x % PatchCountY);
				}

				template<typename TIdx>
				__device__ static TIdx GetResultXY(TIdx& iIdxX, TIdx &iIdxY, TIdx iResultIdx)
				{
					iIdxX = RowSumBaseIdxX() + iResultIdx;
					iIdxY = RowSumBaseIdxY();
					return true;
				}

				template<typename TIdx>
				__device__ static TIdx HasResult(TIdx iResultIdx)
				{
					return true;
				}


				template<typename TSum, int t_iStrideX, int t_iStrideY, typename FuncData, typename FuncResult>
				__device__ static void Sum(TSum* pCache, FuncData funcData, FuncResult funcResult)
				{
					TSum iValue = 0;
					//TSum* pCacheCol = pCache + threadIdx.x * t_iStrideX;

					for (TIdx iIdxY = 0; iIdxY < PatchSizeY; ++iIdxY)
					{
						iValue += funcData(threadIdx.x, iIdxY);
					}

					pCache[threadIdx.x * t_iStrideX] = iValue;

					for (TIdx iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						iValue -= funcData(threadIdx.x, iIdxY - 1);
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1);
						pCache[threadIdx.x * t_iStrideX + iIdxY * t_iStrideY] = iValue;
					}

					__syncthreads();

					const TIdx iSumIdxX = RowSumBaseIdxX();
					const TIdx iSumIdxY = RowSumBaseIdxY();
					const TIdx iStrideY = iSumIdxY * t_iStrideY;

					iValue = pCache[iSumIdxX * t_iStrideX + iStrideY];

					for (TIdx iRelX = 1; iRelX < PatchSizeX; ++iRelX)
					{
						iValue += pCache[(iSumIdxX + iRelX) * t_iStrideX + iStrideY];
					}

					funcResult(iValue, 0, iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < ColGroupSizeX; ++iRelX)
					{
						iValue += pCache[(iSumIdxX + iRelX + PatchSizeX - 1) * t_iStrideX + iStrideY]
							- pCache[(iSumIdxX + iRelX - 1) * t_iStrideX + iStrideY];

						funcResult(iValue, iRelX, iSumIdxX, iSumIdxY);
					}
				}


				template<typename TCache, typename FuncData, typename FuncResult>
				__device__ static void SumCache(TCache& xCache, FuncData funcData, FuncResult funcResult)
				{
					using TSum = typename TCache::TElement;
					TSum iValue = funcData(threadIdx.x, 0);

					for (TIdx iIdxY = 1; iIdxY < PatchSizeY; ++iIdxY)
					{
						iValue += funcData(threadIdx.x, iIdxY);
					}

					xCache.At(threadIdx.x, 0) = iValue;

					for (TIdx iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1)
							- funcData(threadIdx.x, iIdxY - 1);
						xCache.At(threadIdx.x, iIdxY) = iValue;
					}

					__syncthreads();

					const TIdx iSumIdxX = RowSumBaseIdxX();
					const TIdx iSumIdxY = RowSumBaseIdxY();

					iValue = xCache.At(iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < PatchSizeX; ++iRelX)
					{
						iValue += xCache.At(iSumIdxX + iRelX, iSumIdxY);
					}

					funcResult(iValue, 0, iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < ColGroupSizeX; ++iRelX)
					{
						iValue += xCache.At(iSumIdxX + iRelX + PatchSizeX - 1, iSumIdxY)
							- xCache.At(iSumIdxX + iRelX - 1, iSumIdxY);

						funcResult(iValue, iRelX, iSumIdxX, iSumIdxY);
					}
				}

				template<typename TCache, typename FuncData, typename FuncResult>
				__device__ static void SumCache3(TCache& xCache1, TCache& xCache2, TCache& xCache3, FuncData funcData, FuncResult funcResult)
				{
					using TSum = typename TCache::TElement;
					TSum iSum1, iSum2, iSum3;
					TSum iVal1, iVal2, iVal3;

					funcData(iSum1, iSum2, iSum3, threadIdx.x, 0);

					for (TIdx iIdxY = 1; iIdxY < PatchSizeY; ++iIdxY)
					{
						funcData(iVal1, iVal2, iVal3, threadIdx.x, iIdxY);
						iSum1 += iVal1;
						iSum2 += iVal2;
						iSum3 += iVal3;
					}

					xCache1.At(threadIdx.x, 0) = iSum1;
					xCache2.At(threadIdx.x, 0) = iSum2;
					xCache3.At(threadIdx.x, 0) = iSum3;

					for (TIdx iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						funcData(iVal1, iVal2, iVal3, threadIdx.x, iIdxY + PatchSizeY - 1);
						iSum1 += iVal1;
						iSum2 += iVal2;
						iSum3 += iVal3;

						funcData(iVal1, iVal2, iVal3, threadIdx.x, iIdxY - 1);
						iSum1 -= iVal1;
						iSum2 -= iVal2;
						iSum3 -= iVal3;

						xCache1.At(threadIdx.x, iIdxY) = iSum1;
						xCache2.At(threadIdx.x, iIdxY) = iSum2;
						xCache3.At(threadIdx.x, iIdxY) = iSum3;
					}

					__syncthreads();

					const TIdx iSumIdxX = RowSumBaseIdxX();
					const TIdx iSumIdxY = RowSumBaseIdxY();

					iSum1 = xCache1.At(iSumIdxX, iSumIdxY);
					iSum2 = xCache2.At(iSumIdxX, iSumIdxY);
					iSum3 = xCache3.At(iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < PatchSizeX; ++iRelX)
					{
						iSum1 += xCache1.At(iSumIdxX + iRelX, iSumIdxY);
						iSum2 += xCache2.At(iSumIdxX + iRelX, iSumIdxY);
						iSum3 += xCache3.At(iSumIdxX + iRelX, iSumIdxY);
					}

					funcResult(iSum1, iSum2, iSum3, 0, iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < ColGroupSizeX; ++iRelX)
					{
						iSum1 += xCache1.At(iSumIdxX + iRelX + PatchSizeX - 1, iSumIdxY);
						iSum2 += xCache2.At(iSumIdxX + iRelX + PatchSizeX - 1, iSumIdxY);
						iSum3 += xCache3.At(iSumIdxX + iRelX + PatchSizeX - 1, iSumIdxY);

						iSum1 -= xCache1.At(iSumIdxX + iRelX - 1, iSumIdxY);
						iSum2 -= xCache2.At(iSumIdxX + iRelX - 1, iSumIdxY);
						iSum3 -= xCache3.At(iSumIdxX + iRelX - 1, iSumIdxY);

						funcResult(iSum1, iSum2, iSum3, iRelX, iSumIdxX, iSumIdxY);
					}
				}



				template<typename TCache, typename FuncData>
				__device__ static void SumCacheStore(TCache& xCache, FuncData funcData)
				{
					using TSum = typename TCache::TElement;

					TSum iValue = funcData(threadIdx.x, 0);

					for (TIdx iIdxY = 1; iIdxY < PatchSizeY; ++iIdxY)
					{
						iValue += funcData(threadIdx.x, iIdxY);
					}

					xCache.At(threadIdx.x, 0) = iValue;

					for (TIdx iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1)
								- funcData(threadIdx.x, iIdxY - 1);
						xCache.At(threadIdx.x, iIdxY) = iValue;
					}

					__syncthreads();

					const TIdx iSumIdxX = RowSumBaseIdxX() + ColGroupSizeX - 1;
					const TIdx iSumIdxY = RowSumBaseIdxY();


					TSum piStore[PatchSizeX];

					iValue = xCache.At(iSumIdxX, iSumIdxY);
					piStore[0] = iValue;

					for (TIdx iRelX = 1; iRelX < PatchSizeX; ++iRelX)
					{
						piStore[iRelX] = xCache.At(iSumIdxX + iRelX, iSumIdxY);
						iValue += piStore[iRelX];
					}


					xCache.At(iSumIdxX, iSumIdxY) = iValue;

					for (TIdx iRelX = 1; iRelX < ColGroupSizeX; ++iRelX)
					{
						TSum& xStore = piStore[(PatchSizeX - 1) - ((iRelX - 1) % PatchSizeX)];
						iValue -= xStore;

						xStore = xCache.At(iSumIdxX - iRelX, iSumIdxY);

						iValue += xStore;

						xCache.At(iSumIdxX - iRelX, iSumIdxY) = iValue;
					}
				}


			};


		} // namespace Kernel
	} // namespace Cuda
} // namespace Clu

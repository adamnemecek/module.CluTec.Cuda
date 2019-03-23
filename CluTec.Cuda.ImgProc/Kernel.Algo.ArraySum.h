////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Kernel.Algo.ArraySum.h
//
// summary:   Declares the kernel. algo. array sum class
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

			template<int t_iThreadsPerBlockX, int t_iPatchCountY_Pow2, int t_iPatchSizeX, int t_iPatchSizeY>
			struct AlgoArraySum
			{
				using TIdx = short;

				static const int ThreadsPerBlockX = t_iThreadsPerBlockX;
				static const int PatchCountY = 1 << t_iPatchCountY_Pow2;

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
					PRINT(PatchSizeX);
					PRINT(PatchSizeY);
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

				__device__ static int RowSumBaseIdxX()
				{
					return (threadIdx.x / PatchCountY) * ColGroupSizeX;
				}

				__device__ static int RowSumBaseIdxY()
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
					TSum iValue = funcData(threadIdx.x, 0);

					for (TIdx iIdxY = 1; iIdxY < PatchSizeY; ++iIdxY)
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
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1);
						iValue -= funcData(threadIdx.x, iIdxY - 1);
						xCache.At(threadIdx.x, iIdxY) = iValue;
					}

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
						iValue += xCache.At(iSumIdxX + iRelX + PatchSizeX - 1, iSumIdxY);
						iValue -= xCache.At(iSumIdxX + iRelX - 1, iSumIdxY);

						funcResult(iValue, iRelX, iSumIdxX, iSumIdxY);
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
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1);
						iValue -= funcData(threadIdx.x, iIdxY - 1);
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
						TSum xVal = xCache.At(iSumIdxX - iRelX, iSumIdxY);

						iValue += xVal;

						TSum& xStore = piStore[(PatchSizeX - 1) - ((iRelX - 1) % PatchSizeX)];
						iValue -= xStore;

						xStore = xVal;

						xCache.At(iSumIdxX - iRelX, iSumIdxY) = iValue;
					}
				}


				template<typename TCache, typename FuncData, typename FuncStore>
				__device__ static void SumCacheStore(TCache& xCache, FuncData funcData, FuncStore funcStore)
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
						iValue += funcData(threadIdx.x, iIdxY + PatchSizeY - 1);
						iValue -= funcData(threadIdx.x, iIdxY - 1);
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


					xCache.At(iSumIdxX, iSumIdxY) = funcStore(iValue, 0, iSumIdxX, iSumIdxY);

					for (TIdx iRelX = 1; iRelX < ColGroupSizeX; ++iRelX)
					{
						TSum& xStore = piStore[(PatchSizeX - 1) - ((iRelX - 1) % PatchSizeX)];
						iValue -= xStore;

						xStore = xCache.At(iSumIdxX - iRelX, iSumIdxY);

						iValue += xStore;

						xCache.At(iSumIdxX - iRelX, iSumIdxY) = funcStore(iValue, iRelX, iSumIdxX, iSumIdxY);
					}

				}



			};


		} // namespace Kernel
	} // namespace Cuda
} // namespace Clu

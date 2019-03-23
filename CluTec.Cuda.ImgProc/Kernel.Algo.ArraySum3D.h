////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Kernel.Algo.ArraySum3D.h
//
// summary:   Declares the kernel. algo. array sum 3D class
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
			template<int t_iWarpsPerBlockX, int t_iPatchSizeX, int t_iPatchSizeY, int t_iThreadPatchSizeZ>
			struct AlgoArraySum3D_W32
			{
				static_assert(t_iPatchSizeX <= 32, "Patch size in X is larger than 32");

				static const int ThreadsPerWarp = 32;
				static const int WarpsPerBlockX = t_iWarpsPerBlockX;
				static const int ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;
				static const int PatchCountY = 16;

				static const int PatchSizeX = t_iPatchSizeX;
				static const int PatchSizeY = t_iPatchSizeY;
				static const int ThreadPatchSizeZ = t_iThreadPatchSizeZ;
				static const int PatchSizeZ = WarpsPerBlockX * ThreadPatchSizeZ;

				static const int DataArraySizeX = 32;
				static const int DataArraySizeY = PatchCountY + PatchSizeY - 1;

				static const int ColGroupsPerBlock = ThreadsPerBlockX / PatchCountY;
				static const int PatchCountSplitX = ((DataArraySizeX - PatchSizeX + 1) / ColGroupsPerBlock);

				static const int PatchCountX = PatchCountSplitX * ColGroupsPerBlock;

				static const int SumCacheSizeX = ThreadsPerBlockX * ThreadPatchSizeZ;
				static const int SumCacheSizeY = PatchCountY;

#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

				__device__ static void PrintStaticValues()
				{
					PRINT(WarpsPerBlockX);
					PRINT(ThreadsPerBlockX);
					PRINT(PatchCountY);
					PRINT(PatchSizeX);
					PRINT(PatchSizeY);
					PRINT(PatchSizeZ);
					PRINT(ThreadPatchSizeZ);
					PRINT(DataArraySizeX);
					PRINT(DataArraySizeY);
					PRINT(ColGroupsPerBlock);
					PRINT(PatchCountSplitX);
					PRINT(PatchCountX);
					PRINT(SumCacheSizeX);
					PRINT(SumCacheSizeY);
					//PRINT();
					printf("\n");

					__syncthreads();
				}
#undef PRINT

				__device__ __forceinline__ static int ColSumBaseIdxX()
				{
					return threadIdx.x;
				}

				__device__ __forceinline__ static int ColSumBaseIdxY()
				{
					return 0;
				}

				__device__ __forceinline__ static int ColSumBaseIdxZ()
				{
					return 0;
				}

				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int ColSumCacheIdx(int iIdxY, int iIdxZ)
				{
					return (threadIdx.x * ThreadPatchSizeZ + iIdxZ) * t_iStrideX + iIdxY * t_iStrideY;
				}

				__device__ __forceinline__ static int RowSumBaseIdxX()
				{
					// TODO
					return (threadIdx.x / PatchCountY) * PatchCountSplitX;
				}

				__device__ __forceinline__ static int RowSumBaseIdxY()
				{
					return threadIdx.x % PatchCountY;
				}

				__device__ __forceinline__ static int RowSumBaseIdxZ()
				{
					return 0;
				}

				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int RowSumCacheIdx(int iIdxX, int iIdxZ)
				{
					return ((iIdxX + PatchCountSplitX * (threadIdx.x / PatchCountY)) * ThreadPatchSizeZ + iIdxZ) * t_iStrideX 
						+ (threadIdx.x % PatchCountY) * t_iStrideY;
				}

				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int ResultCacheIdx(int iIdxX, int iIdxY, int iIdxZ)
				{
					return (iIdxX * ThreadPatchSizeZ + iIdxZ) * t_iStrideX + iIdxY * t_iStrideY;
				}



				template<typename TSum, int t_iStrideX, int t_iStrideY, typename FuncData, typename FuncResult>
				__device__ static void Sum(TSum* pCache, FuncData funcData, FuncResult funcResult)
				{
					static const int CSX = t_iStrideX;
					static const int CSY = t_iStrideY;

					TSum piValue[ThreadPatchSizeZ];

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = 0;
					}

					// Initial sum over 16 columns, one for each thread in a half-warp.
					// The two different half-warps sum in different z-ranges.
					for (int iIdxY = 0; iIdxY < PatchSizeY; ++iIdxY)
					{
#						pragma unroll
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += funcData(ColSumBaseIdxX()
								, iIdxY
								, ColSumBaseIdxZ() + iIdxZ);
						}
					}

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						pCache[ColSumCacheIdx<CSX, CSY>(0, iIdxZ)] = piValue[iIdxZ];
					}

					for (int iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] -= funcData(ColSumBaseIdxX()
								, iIdxY - 1
								, ColSumBaseIdxZ() + iIdxZ);
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += funcData(ColSumBaseIdxX()
								, iIdxY + PatchSizeY - 1
								, ColSumBaseIdxZ() + iIdxZ);
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							pCache[ColSumCacheIdx<CSX, CSY>(iIdxY, iIdxZ)] = piValue[iIdxZ];
						}
					}


					const int iBaseIdxX = RowSumBaseIdxX();
					const int iBaseIdxY = RowSumBaseIdxY();
					const int iBaseIdxZ = RowSumBaseIdxZ();

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = pCache[RowSumCacheIdx<CSX, CSY>(0, iIdxZ)];
					}

					for (int iIdxX = 1; iIdxX < PatchSizeX; ++iIdxX)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, iIdxZ)];
						}
					}

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						funcResult(piValue[iIdxZ], iBaseIdxX, iBaseIdxY, iBaseIdxZ, iIdxZ);
					}

					for (int iIdxX = 1; iIdxX < PatchCountX; ++iIdxX)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += pCache[RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, iIdxZ)];
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] -= pCache[RowSumCacheIdx<CSX, CSY>(iIdxX - 1, iIdxZ)];
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							funcResult(piValue[iIdxZ], iBaseIdxX, iBaseIdxY, iBaseIdxZ, iIdxZ);
						}
					}
				}



				template<int t_iCount>
				struct Loop
				{
					template<typename FuncOp>
					__device__ __forceinline__ static void Unroll(FuncOp funcOp)
					{
						funcOp(t_iCount - 1);
						Loop<t_iCount - 1>::Unroll(funcOp);
					}
				};

				template<>
				struct Loop<1>
				{
					template<typename FuncOp>
					__device__ __forceinline__ static void Unroll(FuncOp funcOp)
					{
						funcOp(0);
					}
				};

				template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncData>
				__device__ static void SumStore(TSum* pCache, FuncData funcData)
				{
					static const int CSX = t_iStrideX;
					static const int CSY = t_iStrideY;

					TSum piValue[ThreadPatchSizeZ];

					//static const int iIdxZ = 0;

#					pragma unroll
					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = 0;
					}

					// Column sum
					{
						const int iBaseIdxX = ColSumBaseIdxX();
						const int iBaseIdxY = ColSumBaseIdxY();
						const int iBaseIdxZ = ColSumBaseIdxZ();

						// Initial sum over 16 columns, one for each thread in a half-warp.
						// The two different half-warps sum in different z-ranges.
#						pragma unroll
						for (int iIdxY = 0; iIdxY < PatchSizeY; ++iIdxY)
						{
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] += funcData(iBaseIdxX, iIdxY, iBaseIdxZ + iIdxZ);
							}
						}

						//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
						//{
						//	printf("%d: %d [%d]\n", threadIdx.x, ColSumCacheIdx<CSX, CSY>(0, 0), ((ColSumCacheIdx<CSX, CSY>(0, 0) * sizeof(TSum)) / 4) % 32);
						//}

						{
							TSum* const pC = &(pCache[ColSumCacheIdx<CSX, CSY>(0, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								pC[iIdxZ] = piValue[iIdxZ];
							}

							//pC[0] = piValue[0];
							//pC[1] = piValue[1];
							//pC[2] = piValue[2];
							//pC[3] = piValue[3];

							//Loop<ThreadPatchSizeZ>::Unroll([&pC, &piValue](int iIdx)
							//{
							//	pC[iIdx] = piValue[iIdx];
							//});
						}

#						pragma unroll
						for (int iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
						{
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ]
									+= funcData(iBaseIdxX, iIdxY + PatchSizeY - 1, iBaseIdxZ + iIdxZ)
									- funcData(iBaseIdxX, iIdxY - 1, iBaseIdxZ + iIdxZ);
							}

							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = ColSumCacheIdx<CSX, CSY>(iIdxY, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxY, iPos, iBank);
							//}

							{
								TSum* const pC = &(pCache[ColSumCacheIdx<CSX, CSY>(iIdxY, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									pC[iIdxZ] = piValue[iIdxZ];
								}
							}
						}
					}

					__syncthreads();

					{
						const int iBaseIdxX = RowSumBaseIdxX();
						const int iBaseIdxY = RowSumBaseIdxY();
						const int iBaseIdxZ = RowSumBaseIdxZ();

						TSum piStore[ThreadPatchSizeZ];


						{
							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(0, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] = piStore[iIdxZ] = pC[iIdxZ];
							}
						}


#						pragma unroll
						for (int iIdxX = 1; iIdxX < PatchSizeX; ++iIdxX)
						{
							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = RowSumCacheIdx<CSX, CSY>(iIdxX, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxX, iPos, iBank);
							//}

							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] += pC[iIdxZ];
							}
						}

						{
							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(0, 0)]);
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								pC[iIdxZ] = piValue[iIdxZ];
							}
						}

#						pragma unroll
						for (int iIdxX = 1; iIdxX < PatchCountX; ++iIdxX)
						{
							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxX, iPos, iBank);
							//}

							{
								TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									piValue[iIdxZ] += pC[iIdxZ] - piStore[iIdxZ];
								}
							}

							{
								TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									piStore[iIdxZ] = pC[iIdxZ];
									pC[iIdxZ] = piValue[iIdxZ];
								}
							}
						}
					}
				}



				template<typename TCache, typename FuncData, typename FuncResult>
				__device__ static void SumCache(TCache& xCache, FuncData funcData, FuncResult funcResult)
				{
					Sum<TCache::TElement, TCache::StrideX, TCache::StrideY>(xCache.DataPointer(), funcData, funcResult);
				}


			};




			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// Summation of 3D patches with a width of at most 16. Can be used to sum patches for a block
			/// matcher in image space plus disparity space. Summing over the third dimension allows every
			/// thread to perform a number of independent sums, so that ILP can be taken advantage of. ILP:
			/// Instruction Level Parallelism.
			/// </summary>
			///
			/// <typeparam name="t_iWarpsPerBlockX">		The number of warps per block along x-dimension. </typeparam>
			/// <typeparam name="t_iPatchSizeX">			Type of the patch size x coordinate. </typeparam>
			/// <typeparam name="t_iPatchSizeY">			Type of the patch size y coordinate. </typeparam>
			/// <typeparam name="t_iThreadPatchSizeZ">	The number of elements along z-dimension
			/// 											processed by each thread. A number higher than 4
			/// 											will not improve processing speed at current GPUs. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<int t_iWarpsPerBlockX, int t_iPatchSizeX, int t_iPatchSizeY, int t_iThreadPatchSizeZ>
			struct AlgoArraySum3D_W16
			{
				static_assert(t_iPatchSizeX <= 16, "Patch size in X is larger than 16");

				static const int ThreadsPerWarp = 32;
				static const int WarpsPerBlockX = t_iWarpsPerBlockX;
				static const int ThreadsPerBlockX = WarpsPerBlockX * ThreadsPerWarp;
				static const int PatchCountY = 16;

				static const int PatchSizeX = t_iPatchSizeX;
				static const int PatchSizeY = t_iPatchSizeY;
				static const int ThreadPatchSizeZ = t_iThreadPatchSizeZ;
				static const int PatchSizeZ = WarpsPerBlockX * 2 * ThreadPatchSizeZ;

				static const int DataArraySizeX = 16;
				static const int DataArraySizeY = PatchCountY + PatchSizeY - 1;
				
				static const int PatchCountX = 16 - PatchSizeX + 1;

				static const int SumCacheSizeX = ThreadsPerBlockX * ThreadPatchSizeZ;
				static const int SumCacheSizeY = PatchCountY;

				__device__ __forceinline__ static int ColSumBaseIdxX()
				{
					return threadIdx.x % 16;
				}

				__device__ __forceinline__ static int ColSumBaseIdxY()
				{
					return 0;
				}

				__device__ __forceinline__ static int ColSumBaseIdxZ()
				{
					return (threadIdx.x / 16) * ThreadPatchSizeZ;
				}
				
				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int ColSumCacheIdx(int iIdxY, int iIdxZ)
				{
					return (threadIdx.x * ThreadPatchSizeZ + iIdxZ) * t_iStrideX + iIdxY * t_iStrideY;
				}

				__device__ __forceinline__ static int RowSumBaseIdxX()
				{
					return 0;
				}

				__device__ __forceinline__ static int RowSumBaseIdxY()
				{
					return threadIdx.x % 16;
				}

				__device__ __forceinline__ static int RowSumBaseIdxZ()
				{
					return (threadIdx.x / 16) * ThreadPatchSizeZ;
				}

				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int RowSumCacheIdx(int iIdxX, int iIdxZ)
				{
					return ((iIdxX + 16 * (threadIdx.x / 16)) * ThreadPatchSizeZ + iIdxZ) * t_iStrideX + (threadIdx.x % 16) * t_iStrideY;
				}

				template<int t_iStrideX, int t_iStrideY>
				__device__ __forceinline__ static int ResultCacheIdx(int iIdxX, int iIdxY, int iIdxZ)
				{
					return ((iIdxX + 16 * (iIdxZ / ThreadPatchSizeZ)) * ThreadPatchSizeZ + (iIdxZ % ThreadPatchSizeZ)) * t_iStrideX + iIdxY * t_iStrideY;
				}



				template<typename TSum, int t_iStrideX, int t_iStrideY, typename FuncData, typename FuncResult>
				__device__ static void Sum(TSum* pCache, FuncData funcData, FuncResult funcResult)
				{
					static const int CSX = t_iStrideX;
					static const int CSY = t_iStrideY;

					TSum piValue[ThreadPatchSizeZ];

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = 0;
					}

					// Initial sum over 16 columns, one for each thread in a half-warp.
					// The two different half-warps sum in different z-ranges.
					for (int iIdxY = 0; iIdxY < PatchSizeY; ++iIdxY)
					{
#						pragma unroll
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += funcData(ColSumBaseIdxX()
								, iIdxY
								, ColSumBaseIdxZ() + iIdxZ);
						}
					}

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						pCache[ColSumCacheIdx<CSX, CSY>(0, iIdxZ)] = piValue[iIdxZ];
					}

					for (int iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] -= funcData(ColSumBaseIdxX()
								, iIdxY - 1
								, ColSumBaseIdxZ() + iIdxZ);
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += funcData(ColSumBaseIdxX()
								, iIdxY + PatchSizeY - 1
								, ColSumBaseIdxZ() + iIdxZ);
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							pCache[ColSumCacheIdx<CSX, CSY>(iIdxY, iIdxZ)] = piValue[iIdxZ];
						}
					}


					const int iBaseIdxX = RowSumBaseIdxX();
					const int iBaseIdxY = RowSumBaseIdxY();
					const int iBaseIdxZ = RowSumBaseIdxZ();

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = pCache[RowSumCacheIdx<CSX, CSY>(0, iIdxZ)];
					}

					for (int iIdxX = 1; iIdxX < PatchSizeX; ++iIdxX)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, iIdxZ)];
						}
					}

					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						funcResult(piValue[iIdxZ], iBaseIdxX, iBaseIdxY, iBaseIdxZ, iIdxZ);
					}

					for (int iIdxX = 1; iIdxX < PatchCountX; ++iIdxX)
					{
						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] += pCache[RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, iIdxZ)];
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							piValue[iIdxZ] -= pCache[RowSumCacheIdx<CSX, CSY>(iIdxX - 1, iIdxZ)];
						}

						for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
						{
							funcResult(piValue[iIdxZ], iBaseIdxX, iBaseIdxY, iBaseIdxZ, iIdxZ);
						}
					}
				}



				template<int t_iCount>
				struct Loop
				{
					template<typename FuncOp>
					__device__ __forceinline__ static void Unroll(FuncOp funcOp)
					{
						funcOp(t_iCount - 1);
						Loop<t_iCount - 1>::Unroll(funcOp);
					}
				};

				template<>
				struct Loop<1>
				{
					template<typename FuncOp>
					__device__ __forceinline__ static void Unroll(FuncOp funcOp)
					{
						funcOp(0);
					}
				};

				template<int t_iStrideX, int t_iStrideY, typename TSum, typename FuncData>
				__device__ static void SumStore(TSum* pCache, FuncData funcData)
				{
					static const int CSX = t_iStrideX;
					static const int CSY = t_iStrideY;

					TSum piValue[ThreadPatchSizeZ];

					//static const int iIdxZ = 0;

#					pragma unroll
					for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
					{
						piValue[iIdxZ] = 0;
					}

					// Column sum
					{
						const int iBaseIdxX = ColSumBaseIdxX();
						const int iBaseIdxY = ColSumBaseIdxY();
						const int iBaseIdxZ = ColSumBaseIdxZ();

						// Initial sum over 16 columns, one for each thread in a half-warp.
						// The two different half-warps sum in different z-ranges.
#						pragma unroll
						for (int iIdxY = 0; iIdxY < PatchSizeY; ++iIdxY)
						{
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] += funcData(iBaseIdxX, iIdxY, iBaseIdxZ + iIdxZ);
							}
						}

						//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
						//{
						//	printf("%d: %d [%d]\n", threadIdx.x, ColSumCacheIdx<CSX, CSY>(0, 0), ((ColSumCacheIdx<CSX, CSY>(0, 0) * sizeof(TSum)) / 4) % 32);
						//}

						{
							TSum* const pC = &(pCache[ColSumCacheIdx<CSX, CSY>(0, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								pC[iIdxZ] = piValue[iIdxZ];
							}

							//pC[0] = piValue[0];
							//pC[1] = piValue[1];
							//pC[2] = piValue[2];
							//pC[3] = piValue[3];

							//Loop<ThreadPatchSizeZ>::Unroll([&pC, &piValue](int iIdx)
							//{
							//	pC[iIdx] = piValue[iIdx];
							//});
						}

#						pragma unroll
						for (int iIdxY = 1; iIdxY < PatchCountY; ++iIdxY)
						{
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ]
									+= funcData(iBaseIdxX, iIdxY + PatchSizeY - 1, iBaseIdxZ + iIdxZ)
									- funcData(iBaseIdxX, iIdxY - 1, iBaseIdxZ + iIdxZ);
							}

							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = ColSumCacheIdx<CSX, CSY>(iIdxY, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxY, iPos, iBank);
							//}

							{
								TSum* const pC = &(pCache[ColSumCacheIdx<CSX, CSY>(iIdxY, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									pC[iIdxZ] = piValue[iIdxZ];
								}
							}
						}
					}

					__syncthreads();

					{
						const int iBaseIdxX = RowSumBaseIdxX();
						const int iBaseIdxY = RowSumBaseIdxY();
						const int iBaseIdxZ = RowSumBaseIdxZ();

						TSum piStore[ThreadPatchSizeZ];


						{
							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(0, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] = piStore[iIdxZ] = pC[iIdxZ];
							}
						}


#						pragma unroll
						for (int iIdxX = 1; iIdxX < PatchSizeX; ++iIdxX)
						{
							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = RowSumCacheIdx<CSX, CSY>(iIdxX, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxX, iPos, iBank);
							//}

							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, 0)]);

#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								piValue[iIdxZ] += pC[iIdxZ];
							}
						}

						{
							TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(0, 0)]);
#							pragma unroll
							for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
							{
								pC[iIdxZ] = piValue[iIdxZ];
							}
						}

#						pragma unroll
						for (int iIdxX = 1; iIdxX < PatchCountX; ++iIdxX)
						{
							//if (/*threadIdx.x == 0 && threadIdx.y == 0 && */blockIdx.x == 10 && blockIdx.y == 10)
							//{
							//	int iPos = RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, 0);
							//	int iBank = (iPos / 2) % 32;
							//	printf("[%d] %d, %d: %d [%d]\n", CSY, threadIdx.x, iIdxX, iPos, iBank);
							//}

							{
								TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX + PatchSizeX - 1, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									piValue[iIdxZ] += pC[iIdxZ] - piStore[iIdxZ];
								}
							}

							{
								TSum* const pC = &(pCache[RowSumCacheIdx<CSX, CSY>(iIdxX, 0)]);

#								pragma unroll
								for (int iIdxZ = 0; iIdxZ < ThreadPatchSizeZ; ++iIdxZ)
								{
									piStore[iIdxZ] = pC[iIdxZ];
									pC[iIdxZ] = piValue[iIdxZ];
								}
							}
						}
					}
				}



				template<typename TCache, typename FuncData, typename FuncResult>
				__device__ static void SumCache(TCache& xCache, FuncData funcData, FuncResult funcResult)
				{
					Sum<TCache::TElement, TCache::StrideX, TCache::StrideY>(xCache.DataPointer(), funcData, funcResult);
				}


			};


		} // namespace Kernel
	} // namespace Cuda
} // namespace Clu

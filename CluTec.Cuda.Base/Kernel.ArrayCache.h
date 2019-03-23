////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Kernel.ArrayCache.h
//
// summary:   Declares the kernel. array cache class
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

#include "Conversion.h"

#ifdef _DEBUG
#	ifndef DEBUG_BLOCK_X
#		define DEBUG_BLOCK_X 0
#	endif

#	ifndef DEBUG_BLOCK_Y
#		define DEBUG_BLOCK_Y 0
#	endif
#endif

namespace Clu
{
	namespace Cuda
	{
		namespace Kernel
		{

			template<typename _TElement, int t_iWidth_px, int t_iHeight_px, int t_iWarpsPerBlockX, int t_iWarpsPerBlockY,
				int t_iWordsPerRow, int t_iRowWordOffset>
			class CArrayCache
			{
			public:
				using TElement = _TElement;
				using TElementPtr =  TElement*;
				using TElementRef =  TElement&;

				using TIdx = short;

			public:
				static const int c_iWordsPerRow = t_iWordsPerRow;
				// Warps per block
				static const int c_iWarpsPerBlockX = t_iWarpsPerBlockX;
				static const int c_iWarpsPerBlockY = t_iWarpsPerBlockY;

				// Number of words each row is offset
				static const int c_iRowWordOffset = t_iRowWordOffset;

				// Thread per warp
				static const int c_iThreadsPerWarp = 32;

				// Size of the basic unit in the shared cache that can be read by a single thread in one step
				static const int c_iBytesPerCacheUnit = sizeof(TElement);

				// the width of a base patch has to be a full number of words
				static const int c_iWidth_px = t_iWidth_px;
				static const int c_iHeight_px = t_iHeight_px;

				// cache size
				static const int c_iTotalWidth_px = c_iWidth_px;
				static const int c_iTotalWidth_wd = (c_iTotalWidth_px * sizeof(TElement)) / 4 + int((c_iTotalWidth_px * sizeof(TElement)) % 4 > 0);

				// Ensure that the number of words in a row is a multiple of the supplied number. This number should be a power of 2.
				// Also add 1 word to the length of the row to ensure that elements in the same row in consecutive columns are not
				// in the same shared memory bank.
				static const int c_iCacheWidth_wd = (c_iTotalWidth_wd / c_iWordsPerRow + int(c_iTotalWidth_wd % c_iWordsPerRow > 0)) * c_iWordsPerRow + c_iRowWordOffset;
				static const int c_iCacheWidth_px = (c_iCacheWidth_wd * 4) / sizeof(TElement);
				static const int c_iCacheHeight_px = c_iHeight_px;
				static const int c_iChacheSize_px = c_iCacheWidth_px * c_iCacheHeight_px;

				static const int StrideX = 1;
				static const int StrideY = c_iCacheWidth_px;

#define PRINT(theVar) printf(#theVar ": %d\n", theVar)

				__device__ static void PrintStaticValues()
				{
					PRINT(c_iWordsPerRow);
					PRINT(c_iWarpsPerBlockX);
					PRINT(c_iWarpsPerBlockY);
					PRINT(c_iThreadsPerWarp);
					PRINT(c_iBytesPerCacheUnit);
					PRINT(c_iWidth_px);
					PRINT(c_iHeight_px);
					PRINT(c_iTotalWidth_px);
					PRINT(c_iTotalWidth_wd);
					PRINT(c_iCacheWidth_wd);
					PRINT(c_iCacheWidth_px);
					PRINT(c_iCacheHeight_px);
					PRINT(c_iChacheSize_px);
					PRINT(StrideX);
					PRINT(StrideY);
					//PRINT();
					printf("\n");

					__syncthreads();
				}
#undef PRINT

			private:
				 TElement m_pCache[c_iChacheSize_px];


			public:
				__device__ TElementPtr DataPointer()
				{
					return m_pCache;
				}

				__device__ int IsInside(int iX, int iY)
				{
					return int((iX >= 0 && iX < c_iWidth_px) && (iY >= 0 && iY < c_iHeight_px));
				}

				template<int t_iWidth_px, int t_iHeight_px, typename FuncSet>
				__device__ void ForEach(int iStartX_px, int iStartY_px, FuncSet funcSet)
				{
#ifdef CLU_DEBUG_CACHE
					const int iIsDebugThread = (threadIdx.x == 0 && threadIdx.y == 0);
					const int iIsDebugBlock = (blockIdx.x == DEBUG_BLOCK_X && blockIdx.y == DEBUG_BLOCK_Y);
					const int iIsDebug = (iIsDebugThread && iIsDebugBlock);

#endif

					static const int c_iPatchWidth_px = t_iWidth_px;
					static const int c_iPatchHeight_px = t_iHeight_px;

					static const int c_iThreadsPerWidth = c_iWarpsPerBlockX * c_iThreadsPerWarp;
					static const int c_iReadLoopCntX = c_iPatchWidth_px / c_iThreadsPerWidth + int(c_iPatchWidth_px % c_iThreadsPerWidth > 0);
					static const int c_iReadLoopCntY = c_iPatchHeight_px / c_iWarpsPerBlockY + int(c_iPatchHeight_px % c_iWarpsPerBlockY > 0);

#ifdef CLU_DEBUG_CACHE
					if (iIsDebug)
					{
						//printf("Block Idx / Base Pos: %d, %d / %d, %d\n", blockIdx.x, blockIdx.y, iX_px, iY_px);
						printf(
							"c_iWidth_px: %d\n"
							"c_iHeight_px: %d\n"
							"c_iTotalWidth_px: %d\n"
							"c_iTotalWidth_wd: %d\n"
							"c_iCacheWidth_px: %d\n"
							"c_iCacheHeight_px: %d\n"
							"c_iChacheSize_px: %d\n"
							"c_iPatchWidth_px: %d\n"
							"c_iPatchHeight_px: %d\n"
							"c_iThreadsPerWidth: %d\n"
							"c_iReadLoopCntX: %d\n"
							"c_iReadLoopCntY: %d\n"
							, c_iWidth_px
							, c_iHeight_px
							, c_iTotalWidth_px
							, c_iTotalWidth_wd
							, c_iCacheWidth_px
							, c_iCacheHeight_px
							, c_iChacheSize_px
							, c_iPatchWidth_px
							, c_iPatchHeight_px
							, c_iThreadsPerWidth
							, c_iReadLoopCntX
							, c_iReadLoopCntX);
						__syncthreads();
					}
#endif

					for (int iReadLoopIdxY = 0; iReadLoopIdxY < c_iReadLoopCntY; ++iReadLoopIdxY)
					{
						const int iImgY_px = iStartY_px + threadIdx.y + c_iWarpsPerBlockY * iReadLoopIdxY;
						const int iCacheY_px = threadIdx.y + c_iWarpsPerBlockY * iReadLoopIdxY;
						const int iCacheStrideY_px = iCacheY_px * c_iCacheWidth_px;

						if (iCacheY_px >= c_iPatchHeight_px)
						{
							break;
						}

						for (int iReadLoopIdxX = 0; iReadLoopIdxX < c_iReadLoopCntX; ++iReadLoopIdxX)
						{
							const int iImgX_px = iStartX_px + threadIdx.x + c_iThreadsPerWidth * iReadLoopIdxX;
							const int iCacheX_px = threadIdx.x + c_iThreadsPerWidth * iReadLoopIdxX;

							const int iCachePos_px = iCacheStrideY_px + iCacheX_px;

							if (iCacheX_px >= c_iPatchWidth_px)
							{
								break;
							}

							funcSet(m_pCache[iCachePos_px], iImgX_px, iImgY_px, iCacheX_px, iCacheY_px);
						}
					}
				}


				template<int t_iHeight_px, typename FuncSet>
				__device__ void ForEachVarWidth(int iStartX_px, int iStartY_px, int iWidth_px, FuncSet funcSet)
				{
					int iPatchWidth_px = iWidth_px;
					static const int c_iPatchHeight_px = t_iHeight_px;

					static const int c_iThreadsPerWidth = c_iWarpsPerBlockX * c_iThreadsPerWarp;
					int iReadLoopCntX = iPatchWidth_px / c_iThreadsPerWidth + int(iPatchWidth_px % c_iThreadsPerWidth > 0);
					static const int c_iReadLoopCntY = c_iPatchHeight_px / c_iWarpsPerBlockY + int(c_iPatchHeight_px % c_iWarpsPerBlockY > 0);

					for (int iReadLoopIdxY = 0; iReadLoopIdxY < c_iReadLoopCntY; ++iReadLoopIdxY)
					{
						const int iImgY_px = iStartY_px + threadIdx.y + c_iWarpsPerBlockY * iReadLoopIdxY;
						const int iCacheY_px = threadIdx.y + c_iWarpsPerBlockY * iReadLoopIdxY;
						const int iCacheStrideY_px = iCacheY_px * c_iCacheWidth_px;

						if (iCacheY_px >= c_iPatchHeight_px)
						{
							break;
						}

						for (int iReadLoopIdxX = 0; iReadLoopIdxX < iReadLoopCntX; ++iReadLoopIdxX)
						{
							const int iCacheX_px = threadIdx.x + c_iThreadsPerWidth * iReadLoopIdxX;
							const int iImgX_px = iStartX_px + iCacheX_px;

							const int iCachePos_px = iCacheStrideY_px + iCacheX_px;

							if (iCacheX_px >= iPatchWidth_px)
							{
								break;
							}
							////if (threadIdx.x == 0 && threadIdx.y == 0)
							//{
							//	printf("%d, %d > %d: %d, %d, %d, %d\n", blockIdx.x, blockIdx.y, iPatchWidth_px, iImgX_px, iImgY_px, iCacheX_px, iCacheY_px);
							//}
							funcSet(m_pCache[iCachePos_px], iImgX_px, iImgY_px, iCacheX_px, iCacheY_px);
						}
					}
				}



				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Reads from surf. </summary>
				///
				/// <typeparam name="t_iWidth_px"> 	Type of the i width px. </typeparam>
				/// <typeparam name="t_iHeight_px">	Type of the i height px. </typeparam>
				/// <param name="surfImg">   	The surf image. </param>
				/// <param name="iStartX_px">	The start x coordinate px. </param>
				/// <param name="iStartY_px">	The start y coordinate px. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<int t_iWidth_px, int t_iHeight_px>
				__device__ void ReadFromSurf(const Clu::Cuda::_CDeviceSurface& surfImg
					, int iStartX_px, int iStartY_px)
				{
					ForEach<t_iWidth_px, t_iHeight_px>(iStartX_px, iStartY_px, [&surfImg](TElementRef xValue, int iImgX_px, int iImgY_px, int iCacheX, int iCacheY)
					{
						xValue = surfImg.Read2D<TElement>(iImgX_px, iImgY_px);
					});

				}

				template<int t_iWidth_px, int t_iHeight_px>
				__device__ void SafeReadFromSurf(const Clu::Cuda::_CDeviceSurface& surfImg
					, int iStartX_px, int iStartY_px)
				{
					ForEach<t_iWidth_px, t_iHeight_px>(iStartX_px, iStartY_px, [&surfImg](TElementRef xValue, int iImgX_px, int iImgY_px, int iCacheX, int iCacheY)
					{
						if (surfImg.IsInside(iImgX_px, iImgY_px))
						{
							xValue = surfImg.Read2D<TElement>(iImgX_px, iImgY_px);
						}
						else
						{
							xValue = Clu::Cuda::MakeZero<TElement>();
						}
					});

				}


				template<int t_iHeight_px>
				__device__ void ReadFromSurfVarWidth(const Clu::Cuda::_CDeviceSurface& surfImg
					, int iStartX_px, int iStartY_px, int iWidth_px)
				{
					ForEachVarWidth<t_iHeight_px>(iStartX_px, iStartY_px, iWidth_px
						, [&surfImg](TElementRef xValue, int iImgX_px, int iImgY_px, int iCacheX, int iCacheY)
					{
						xValue = surfImg.Read2D<TElement>(iImgX_px, iImgY_px);
						//Clu::Cuda::Assign(xValue, surfImg.Read2D<TElement>(iImgX_px, iImgY_px));
					});

				}


				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Writes to surf. </summary>
				///
				/// <typeparam name="t_iWidth_px"> 	Type of the i width px. </typeparam>
				/// <typeparam name="t_iHeight_px">	Type of the i height px. </typeparam>
				/// <param name="surfImg">   	The surf image. </param>
				/// <param name="iStartX_px">	The start x coordinate px. </param>
				/// <param name="iStartY_px">	The start y coordinate px. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<int t_iWidth_px, int t_iHeight_px, typename FuncWrite>//, typename _TElementTrg>
				__device__ void WriteToSurf(Clu::Cuda::_CDeviceSurface& surfImg
					, int iStartX_px, int iStartY_px, FuncWrite funcWrite)
				{
					ForEach<t_iWidth_px, t_iHeight_px>(iStartX_px, iStartY_px, [&surfImg, funcWrite](TElementRef xValue, int iImgX_px, int iImgY_px, int iCacheX_px, int iCacheY_px)
					{
						funcWrite(xValue, surfImg, iImgX_px, iImgY_px, iCacheX_px, iCacheY_px);
					});
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Returns a cache unit starting at the given pixel. </summary>
				///
				/// <param name="iX_px">	Zero-based index of the x coordinate px. </param>
				/// <param name="iY_px">	Zero-based index of the y coordinate px. </param>
				///
				/// <returns>	A TElement&amp; </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename _TIdx1, typename _TIdx2>
				__device__ __forceinline__ TElementRef At(_TIdx1 iX_px, _TIdx2 iY_px)
				{
					const int iPos_px = int(iY_px) * c_iCacheWidth_px + int(iX_px);

					return m_pCache[iPos_px];
				}


			};
		} // namespace Kernel
	} // namespace Cuda
} // namespace Clu
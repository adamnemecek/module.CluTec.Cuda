////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Kernel.ArrayCache_W16.h
//
// summary:   Declares the kernel. array cache w 16 class
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
			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// An array cache optimized for a width of 16 words per row. For example, if you have 32
			/// elements of type short per row, than these are 16 words. The cache organizes consecutive
			/// lines in shared memory banks 0-15 and 16-31. Furthermore, every two rows the column origin is
			/// moved by two(!) columns. This has the effect that one half-warp (16 threads) can process a 16
			/// row by 32 columns block starting at an even bank and the other half warp can process a block
			/// of the same size starting at an odd bank.
			/// </summary>
			///
			/// <typeparam name="_TElement">			Type of the element. </typeparam>
			/// <typeparam name="t_iWidth_px">			Type of the i width px. </typeparam>
			/// <typeparam name="t_iHeight_px">			Type of the i height px. </typeparam>
			/// <typeparam name="t_iSubCacheCntPow">	Type of the sub cache count pow. </typeparam>
			/// <typeparam name="t_iWarpsPerBlockX">	Type of the warps per block x coordinate. </typeparam>
			/// <typeparam name="t_iWarpsPerBlockY">	Type of the warps per block y coordinate. </typeparam>
			/// <typeparam name="t_iWordsPerRow">   	Type of the words per row. </typeparam>
			/// <typeparam name="t_iRowWordOffset"> 	Type of the row word offset. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<typename _TElement>
			class CArrayCache_W16
			{
			public:
				using TElement = _TElement;
				using TElementPtr =  TElement*;
				using TElementRef =  TElement&;

			public:
				static const int c_iWordsPerRow = 16;
				// Warps per block
				static const int c_iWarpsPerBlockX = 1;
				static const int c_iWarpsPerBlockY = 1;

				// Thread per warp
				static const int c_iThreadsPerWarp = 32;

				// Size of the basic unit in the shared cache that can be read by a single thread in one step
				static const int c_iBytesPerCacheUnit = sizeof(TElement);
				static const int c_iPixelPerWord = 4 / c_iBytesPerCacheUnit;

				// the width of a base patch has to be a full number of words
				static const int c_iWidth_px = c_iWordsPerRow * c_iPixelPerWord;
				static const int c_iHeight_px = 16;

				// cache size
				static const int c_iTotalWidth_px = c_iWidth_px;
				static const int c_iTotalWidth_wd = c_iHeight_px;

				static const int c_iCacheWidth_wd = c_iWordsPerRow;
				static const int c_iCacheWidth_px = c_iWidth_px;
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

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Index at. </summary>
				///
				/// <param name="iX_px">	Zero-based index of the x coordinate px. </param>
				/// <param name="iY_px">	Zero-based index of the y coordinate px. </param>
				///
				/// <returns>	An int. </returns>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename _TIdx1, typename _TIdx2>
				__device__ __forceinline__ static int IndexAt(_TIdx1 iX_px, _TIdx2 iY_px)
				{
					//if (iX_px < 0 || iX_px > c_iWidth_px
					//	|| iY_px < 0 || iY_px > c_iHeight_px)
					//{
					//	printf("Index out of range: %d, %d\n", iX_px, iY_px);
					//}

					const int iPosY_wd = int(iY_px) * c_iWordsPerRow + (int(iY_px) >> 1) * 2;
					return iPosY_wd * c_iPixelPerWord + iX_px;
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
					return m_pCache[IndexAt(iX_px, iY_px)];
				}


			};
		} // namespace Kernel
	} // namespace Cuda
} // namespace Clu
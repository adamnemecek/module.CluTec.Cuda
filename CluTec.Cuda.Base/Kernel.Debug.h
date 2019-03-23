////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Kernel.Debug.h
//
// summary:   Declares the kernel. debug class
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

#include "CluTec.Types1/Defines.h"
#include "Defines.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Kernel
		{
			namespace Debug
			{
				template<typename FuncDebug>
				__CUDA_DI__ void Run(FuncDebug funcDebug)
				{
#ifdef CLU_DEBUG_KERNEL
					funcDebug();
#endif
				}

				__CUDA_DI__ int IsThread(int iX, int iY)
				{
					return (threadIdx.x == iX && threadIdx.y == iY);
				}

				__CUDA_DI__ int IsBlock(int iX, int iY)
				{
					return (blockIdx.x == iX && blockIdx.y == iY);
				}

				__CUDA_DI__ int IsThreadAndBlock(int iTX, int iTY, int iBX, int iBY)
				{
					return (IsThread(iTX, iTY) > 0) && (IsBlock(iBX, iBY) > 0);
				}

			}
		}
	}
}
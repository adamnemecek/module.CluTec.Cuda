////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      KernelDriverBase.cpp
//
// summary:   Implements the kernel driver base class
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

#include <cmath>

#include "KernelDriverBase.h"
#include "CluTec.Base/Exception.h"

namespace Clu
{
	namespace Cuda
	{
		void CKernelDriverBase::_SetKernelConfigDims(const dim3& dimBlocksInGrid, const dim3& dimThreadsPerBlock, TId uKernelId)
		{
			if (uKernelId >= m_vecDimBlocksInGrid.size()
				|| uKernelId >= m_vecDimThreadsPerBlock.size())
			{
				m_vecDimBlocksInGrid.resize(uKernelId + 1);
				m_vecDimThreadsPerBlock.resize(uKernelId + 1);
			}

			m_vecDimBlocksInGrid[uKernelId] = dimBlocksInGrid;
			m_vecDimThreadsPerBlock[uKernelId] = dimThreadsPerBlock;
		}

		void CKernelDriverBase::EvalThreadConfig(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
			TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY, TSize nAlgoRegCnt, TId uKernelId)
		{
			if (!xFormat.IsValid())
			{
				throw CLU_EXCEPTION("Invalid image format");
			}

			// Get the maximal number of registers per block
			TSize nMaxRegsPerBlock = xDevice.NumberOf(EDeviceProperty::RegistersPerBlock);

			TSize nThreadsPerBlock = 0;
			TSize nWarpsPerBlock = nDesiredWarpsPerBlockX * nDesiredWarpsPerBlockY;

			if (nWarpsPerBlock == 0)
			{
				TSize nMaxWarps = (TSize)std::floor(double(nMaxRegsPerBlock) / double(nAlgoRegCnt * ThreadsPerWarp));
				nThreadsPerBlock = nMaxWarps * ThreadsPerWarp;
				nWarpsPerBlock = nMaxWarps;
			}
			else
			{
				// The desired threads per block
				nThreadsPerBlock = nWarpsPerBlock * ThreadsPerWarp;

				// Reduce threads per block until the required number of register per block is less then its maximum
				if (nThreadsPerBlock * nAlgoRegCnt > nMaxRegsPerBlock)
				{
					nWarpsPerBlock = (TSize)std::floor(double(nMaxRegsPerBlock) / double(nAlgoRegCnt * ThreadsPerWarp));
					nThreadsPerBlock = nWarpsPerBlock * ThreadsPerWarp;
				}
			}

			if (nThreadsPerBlock > xDevice.NumberOf(EDeviceProperty::ThreadsPerBlock))
			{
				nWarpsPerBlock = xDevice.NumberOf(EDeviceProperty::ThreadsPerBlock) / ThreadsPerWarp;
			}

			TSize nWarpsPerWidth = xFormat.iWidth / ThreadsPerWarp + (xFormat.iWidth % ThreadsPerWarp > 0 ? 1 : 0);

			TSize nWarps = nWarpsPerWidth;
			TSize nBlockCntX = 1;
			while (nWarpsPerBlock < nWarps)
			{
				++nBlockCntX;
				nWarps = nWarpsPerWidth / nBlockCntX + (nWarpsPerWidth % nBlockCntX > 0 ? 1 : 0);
			}

			nWarpsPerBlock = nWarps;

			TSize nWarpsPerBlockY = (nDesiredWarpsPerBlockY == 0 ? 1 : nDesiredWarpsPerBlockY);
			TSize nWarpsPerBlockX = nWarpsPerBlock / nWarpsPerBlockY;

			if (nWarpsPerBlockX == 0 && nWarpsPerBlockY <= 1)
			{
				throw CLU_EXCEPTION("Something went wrong in kernel configuration");
			}

			while (nWarpsPerBlockX == 0 && nWarpsPerBlockY > 1)
			{
				--nWarpsPerBlockY;
				nWarpsPerBlockX = nWarpsPerBlock / nWarpsPerBlockY;
			}

			nWarpsPerBlock = nWarpsPerBlockX * nWarpsPerBlockY;
			nThreadsPerBlock = nWarpsPerBlock * ThreadsPerWarp;

			TSize nThreadsPerBlockX = nWarpsPerBlockX * ThreadsPerWarp;

			nBlockCntX = nWarpsPerWidth / nWarpsPerBlockX + (nWarpsPerWidth % nWarpsPerBlockX > 0 ? 1 : 0);

			TSize nBlockCntY = TSize(xFormat.iHeight) / nWarpsPerBlockY + (TSize(xFormat.iHeight) % nWarpsPerBlockY > 0 ? 1 : 0);

			_SetKernelConfigDims(
				dim3(nBlockCntX, nBlockCntY, 1)
				, dim3((int)nThreadsPerBlockX, (int)nWarpsPerBlockY, 1)
				, uKernelId);
		}



		void CKernelDriverBase::EvalThreadConfigBlockSize(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
			TSize nBlockPixelWidth, TSize nBlockPixelHeight,
			TSize nOffsetLeft, TSize nOffsetRight, TSize nOffsetTop, TSize nOffsetBottom,
			TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY,
			TSize nAlgoRegCnt, bool bOnlyFullBlocks, TId uKernelId)
		{

			if (!xFormat.IsValid())
			{
				throw CLU_EXCEPTION("Invalid image format");
			}

			TSize nEffWidth = TSize(xFormat.iWidth) - nOffsetLeft - nOffsetRight;
			TSize nEffHeight = TSize(xFormat.iHeight) - nOffsetBottom - nOffsetTop;

			TSize nBlockCntX = nEffWidth / nBlockPixelWidth;
			TSize nBlockCntY = nEffHeight / nBlockPixelHeight;

			if (!bOnlyFullBlocks)
			{
				nBlockCntX += (nEffWidth % nBlockPixelWidth > 0 ? 1 : 0);
				nBlockCntY += (nEffHeight % nBlockPixelHeight > 0 ? 1 : 0);
			}

			TSize nThreadsPerBlockX = nDesiredWarpsPerBlockX * ThreadsPerWarp;
			TSize nThreadsPerBlockY = nDesiredWarpsPerBlockY;

			_SetKernelConfigDims(
				dim3(nBlockCntX, nBlockCntY, 1)
				, dim3(nThreadsPerBlockX, nThreadsPerBlockY, 1)
				, uKernelId);
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////

		void CKernelDriverBase::EvalThreadConfig_ReadDepBlockSize(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
			TSize nResultPixelWidth, TSize nResultPixelHeight,
			TSize nReadPixelWidth, TSize nReadPixelHeight,
			TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY,
			TSize nAlgoRegCnt, bool bOnlyFullBlocks, TId uKernelId)
		{

			if (!xFormat.IsValid())
			{
				throw CLU_EXCEPTION("Invalid image format");
			}

			TSize nEffWidth = TSize(xFormat.iWidth) - (nReadPixelWidth - nResultPixelWidth);
			TSize nEffHeight = TSize(xFormat.iHeight) - (nReadPixelHeight - nResultPixelHeight);

			TSize nBlockCntX = nEffWidth / nResultPixelWidth;
			TSize nBlockCntY = nEffHeight / nResultPixelHeight;

			if (!bOnlyFullBlocks)
			{
				nBlockCntX += (nEffWidth % nResultPixelWidth > 0 ? 1 : 0);
				nBlockCntY += (nEffHeight % nResultPixelHeight > 0 ? 1 : 0);
			}

			TSize nThreadsPerBlockX = nDesiredWarpsPerBlockX * ThreadsPerWarp;
			TSize nThreadsPerBlockY = nDesiredWarpsPerBlockY;

			_SetKernelConfigDims(
				dim3(nBlockCntX, nBlockCntY, 1)
				, dim3(nThreadsPerBlockX, nThreadsPerBlockY, 1)
				, uKernelId);
		}



	}
}

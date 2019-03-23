////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      KernelDriverBase.h
//
// summary:   Declares the kernel driver base class
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
#include <vector>

#include "CluTec.Types1/ImageFormat.h"
#include "CluTec.Base/Logger.h"
#include "CluTec.Base/Exception.h"

#include "Api.h"
#include "DeviceEvent.h"
#include "Device.h"


#define CLU_KERNEL_CONFIG() <<<BlocksInGrid(CKernelDriverBase::DefaultKernel), ThreadsPerBlock(CKernelDriverBase::DefaultKernel)>>>
#define CLU_KERNEL_CONFIG_(theKernelId) <<<BlocksInGrid(theKernelId), ThreadsPerBlock(theKernelId)>>>

namespace Clu
{
	namespace Cuda
	{

		class CKernelDriverBase
		{
		public:
			using TSize = unsigned;
			using TId = unsigned;

		public:
			static const TSize ThreadsPerWarp = 32;
			static const TId DefaultKernel = 0;

		private:
			std::vector<dim3> m_vecDimBlocksInGrid;
			std::vector<dim3> m_vecDimThreadsPerBlock;

		protected:
			CDeviceEvent m_evStart, m_evStop;
			double m_dLastProcessTime;

			Clu::CIString m_sKernelName;

		protected:
			void _SetKernelConfigDims(const dim3& dimBlocksInGrid, const dim3& dimThreadsPerBlock, TId uKernelId);

		public:
			CKernelDriverBase(const char* pcKernelName)
			{
				m_vecDimBlocksInGrid.emplace_back(dim3(1, 1, 1));
				m_vecDimThreadsPerBlock.emplace_back(dim3(1, 1, 1));
				m_dLastProcessTime = 0.0;
				m_sKernelName = pcKernelName;
			}

			~CKernelDriverBase()
			{}

			void EvalThreadConfig(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
				TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY, TSize nAlgoRegCnt, TId uKernelId = DefaultKernel);

			void EvalThreadConfigBlockSize(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
				TSize nBlockPixelWidth, TSize nBlockPixelHeight,
				TSize nOffsetLeft, TSize nOffsetRight, TSize nOffsetTop, TSize nOffsetBottom,
				TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY,
				TSize nAlgoRegCnt, bool bOnlyFullBlocks = true, TId uKernelId = DefaultKernel);

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// Evaluate the kernel configuration based on the two block sizes: 1. The block size of pixels
			/// for which a result is calculated, and 2. The block size of the read pixels that are needed to
			/// calculate the results.
			/// 
			/// For example, in a convolution with a filter size W each convolution only generates the filter
			/// result for a single pixel. However, an area of WxW pixels have to be read to generate the
			/// result.
			/// </summary>
			///
			/// <param name="xDevice">				 	The device. </param>
			/// <param name="xFormat">				 	Describes the format to use. </param>
			/// <param name="nResultPixelWidth">	 	Width of the block pixel. </param>
			/// <param name="nResultPixelHeight">	 	Height of the block pixel. </param>
			/// <param name="nResultToReadOffsetX">  	The offset left. </param>
			/// <param name="nResultToReadOffsetY">  	The offset right. </param>
			/// <param name="nReadPixelWidth">		 	The offset top. </param>
			/// <param name="nReadPixelHeight">		 	The offset bottom. </param>
			/// <param name="nDesiredWarpsPerBlockX">	The desired warps per block x coordinate. </param>
			/// <param name="nDesiredWarpsPerBlockY">	The desired warps per block y coordinate. </param>
			/// <param name="nAlgoRegCnt">			 	Number of algo registers. </param>
			/// <param name="bOnlyFullBlocks">		 	true to only full blocks. </param>
			/// <param name="uKernelId">			 	Identifier for the kernel. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			void EvalThreadConfig_ReadDepBlockSize(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
				TSize nResultPixelWidth, TSize nResultPixelHeight,
				TSize nReadPixelWidth, TSize nReadPixelHeight,
				TSize nDesiredWarpsPerBlockX, TSize nDesiredWarpsPerBlockY,
				TSize nAlgoRegCnt, bool bOnlyFullBlocks = true, TId uKernelId = DefaultKernel);


			const Clu::CIString& KernelName() const
			{
				return m_sKernelName;
			}

			const dim3& BlocksInGrid(TId uId = DefaultKernel) const
			{
				return m_vecDimBlocksInGrid[uId];
			}

			const dim3& ThreadsPerBlock(TId uId = DefaultKernel) const
			{
				return m_vecDimThreadsPerBlock[uId];
			}

			void Begin()
			{
				m_evStart.Record();
			}

			void End() 
			{
				m_evStop.RecordAndSync();
				m_dLastProcessTime = Cuda::EventElapsedTime(m_evStart, m_evStop);
			}

			void SafeBegin(const Clu::CIString& sName)
			{
				try
				{
					Begin();
				}
				CLU_CATCH_RETHROW_ALL(CLU_S "Error before starting kernel '" << sName << "'");
			}

			void SafeEnd(const Clu::CIString& sName)
			{
				try
				{
					End();
				}
				CLU_CATCH_RETHROW_ALL(CLU_S "Error after running kernel '" << sName << "'");
			}

			double LastProcessTime()
			{
				return m_dLastProcessTime;
			}

			void LogLastProcessTime(const Clu::CIString& sName)
			{
				CLU_LOG(CLU_S "Kernel '" << sName << "': " << LastProcessTime() << "ms");
			}
		};


		template<typename TDriver, typename... TPars>
		double ProcessKernel(TDriver &xDriver, TPars&&... xPars)
		{
			try
			{
				xDriver.SafeBegin(xDriver.KernelName());

				try
				{
					xDriver.Process(std::forward<TPars>(xPars)...);
				}
				CLU_CATCH_RETHROW_ALL(CLU_S "Error while running kernel '" << xDriver.KernelName() << "'");

				xDriver.SafeEnd(xDriver.KernelName());

				xDriver.LogLastProcessTime(xDriver.KernelName());
			}
			CLU_CATCH_RETHROW_ALL(CLU_S "Error processing kernel '" << xDriver.KernelName() << "'");
			
			return xDriver.LastProcessTime();
		}



	}
}

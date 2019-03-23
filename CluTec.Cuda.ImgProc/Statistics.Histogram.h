////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Statistics.Histogram.h
//
// summary:   Declares the statistics. histogram class
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

#include "cuda_runtime.h"
#include "CluTec.Types1/IArrayInt32.h"
#include "CluTec.Types1/IArrayInt64.h"
#include "CluTec.Math/Static.Matrix.h"
#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.ImgProc/Camera.Pinhole.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Statistics
		{
			namespace Histogram
			{
				using TPixelHist = TPixel_RGBA_UInt32;
				using TPixelValueRange = TPixel_RGBA_Double;

				struct _SParameter
				{
					TPixelValueRange pixMin;
					TPixelValueRange pixMax;
					unsigned uBucketCount;
				};


				struct SParameter : public _SParameter
				{
					// If this value is -1, then evaluate histogram for all channels.
					// Otherwise, calculate histogram for the given channel.
					int iSingleChannel;

					void Set(TPixelValueRange _pixMin, TPixelValueRange _pixMax, unsigned _uBucketCount
					, int _iSingleChannel)
					{
						pixMin = _pixMin;
						pixMax = _pixMax;
						uBucketCount = _uBucketCount;
						iSingleChannel = _iSingleChannel;
					}
				};

				class CDriver : public CKernelDriverBase
				{
				public:
#ifdef _DEBUG
					static const size_t NumberOfRegisters = 30;
#else
					static const size_t NumberOfRegisters = 30;
#endif

				protected:
					// The number of image channels processed 
					uint32_t m_uChannelCount;
					TPixelHist m_pixTotalCount;

					SParameter m_xPars;
					Clu::CIImage m_imgHist;
					Clu::Cuda::CDeviceImage m_deviHist;

					void _SelectPixelType(const Clu::Cuda::_CDeviceSurface& surfImage);

					template<typename TPixelType>
					void _SelectDataType(const Clu::Cuda::_CDeviceSurface& surfImage);

					template<typename TPixelType, typename TDataType>
					void _SelectAlgo(const Clu::Cuda::_CDeviceSurface& surfImage);

					double _GetValueAtPercentCount(double dPercent, double dValueMin, double dValueMax, double dTotalCount, unsigned uChannel);

					void _GetValueAtPercentCount(double& dResultMin, double& dResultMax, double dPercentMin, double dPercentMax
						, double dValueMin, double dValueMax, double dTotalCount, unsigned uChannel);

				public:
					CDriver()
						: CKernelDriverBase("Histogram")
					{
						m_uChannelCount = 0;
						m_pixTotalCount.SetZero();
					}

					~CDriver()
					{}

					void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat, const SParameter& xPars);

					double Run(const Clu::Cuda::_CDeviceSurface& surfImage);

					bool HasData()
					{
						return m_uChannelCount > 0;
					}

					uint32_t ChannelCount() const
					{
						return m_uChannelCount;
					}

					bool IsSingleChannel() const
					{
						return m_xPars.iSingleChannel >= 0;
					}

					int SingleChannel() const
					{
						return m_xPars.iSingleChannel;
					}

					template<typename TArray>
					void GetData(TArray& aData, unsigned uChannel);

					TPixelValueRange GetValueAtPercentCount(double dPercent);
					double GetValueAtPercentCount(double dPercent, unsigned uChannel);
					void GetValueAtPercentCount(double& dValueMin, double& dValueMax, double dPercentMin, double dPercentMax, unsigned uChannel);

				};
			} // Histogram
		} // Statistics
	} // Cuda
} // Clu


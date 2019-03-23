////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Statistics.Histogram.cu
//
// summary:   statistics. histogram class
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

#include "cuda_runtime.h"

#include "Statistics.Histogram.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

//#define CLU_DEBUG_KERNEL
//#include "CluTec.Cuda.Base/Kernel.Debug.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Statistics
		{
			namespace Histogram
			{
				namespace Kernel
				{
					struct Const
					{
						static const int WarpsPerBlockX = 4;
						static const int WarpsPerBlockY = 1;
						static const int ThreadCountX = 8;
						static const int ThreadCountY = 16;
						static const int BlockSizeX = ThreadCountX;
						static const int BlockSizeY = ThreadCountY;
					};



					template<typename TPixel, uint32_t t_uChannelCount>
					struct SAlgo
					{
						__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceImage deviHist, Clu::Cuda::_CDeviceSurface surfImage, _SParameter xPars)
						{
							static const uint32_t ChannelCount = t_uChannelCount;
							using TData = typename TPixel::TData;

							int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
							int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
							if (!surfImage.IsInside(nSrcX, nSrcY))
							{
								return;
							}

							TPixel pixValue = surfImage.ReadPixel2D<TPixel>(nSrcX, nSrcY);

							double pdBucketIdx[ChannelCount];
							// Use separate loop here so that it may be parallelized via instruction level parallelism.
							for (uint32_t uChannel = 0; uChannel < ChannelCount; ++uChannel)
							{
								pdBucketIdx[uChannel] = floor((double(pixValue[uChannel]) - double(xPars.pixMin[uChannel])) 
										/ (double(xPars.pixMax[uChannel]) - double(xPars.pixMin[uChannel])) * double(xPars.uBucketCount));
							}

							TPixelHist* pHist = (TPixelHist*)deviHist.DataPointer();

							for (uint32_t uChannel = 0; uChannel < ChannelCount; ++uChannel)
							{
								double dBucketIdx = pdBucketIdx[uChannel];

								if (dBucketIdx >= 0.0 && dBucketIdx <= double(xPars.uBucketCount - 1))
								{
									atomicAdd(&(pHist[unsigned(dBucketIdx)][uChannel]), 1);
								}
							}
						}

						template<uint32_t t_uChannel>
						__device__ __forceinline__ static void ProcessChannel(Clu::Cuda::_CDeviceImage deviHist, Clu::Cuda::_CDeviceSurface surfImage, _SParameter xPars)
						{
							static const uint32_t Channel = t_uChannel;
							using TData = typename TPixel::TData;

							int nSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
							int nSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
							if (!surfImage.IsInside(nSrcX, nSrcY))
							{
								return;
							}

							TPixel pixValue = surfImage.ReadPixel2D<TPixel>(nSrcX, nSrcY);

							double dBucketIdx = floor((double(pixValue[Channel]) - double(xPars.pixMin[Channel]))
								/ (double(xPars.pixMax[Channel]) - double(xPars.pixMin[Channel])) * double(xPars.uBucketCount));

							TPixelHist* pHist = (TPixelHist*)deviHist.DataPointer();

							if (dBucketIdx >= 0.0 && dBucketIdx <= double(xPars.uBucketCount - 1))
							{
								atomicAdd(&(pHist[unsigned(dBucketIdx)][Channel]), 1);
							}
						}
					};



					template<typename TPixelType, typename TDataType>
					__global__ void Process(Clu::Cuda::_CDeviceImage deviHist, Clu::Cuda::_CDeviceSurface surfImage, _SParameter xPars)
					{
						using TPixel = SPixel<TPixelType, TDataType>;

						SAlgo<TPixel, TPixelType::ChannelCount>::Process(deviHist, surfImage, xPars);
					}

					template<typename TPixelType, typename TDataType, uint32_t t_Channel>
					__global__ void ProcessChannel(Clu::Cuda::_CDeviceImage deviHist, Clu::Cuda::_CDeviceSurface surfImage, _SParameter xPars)
					{
						using TPixel = SPixel<TPixelType, TDataType>;

						SAlgo<TPixel, TPixelType::ChannelCount>::ProcessChannel<t_Channel>(deviHist, surfImage, xPars);
					}

				}

				template<typename TPixelType, typename TDataType>
				void CDriver::_SelectAlgo(const Clu::Cuda::_CDeviceSurface& surfImage)
				{
					if (m_xPars.iSingleChannel < 0)
					{
						Kernel::Process<TPixelType, TDataType>
							CLU_KERNEL_CONFIG()
							(m_deviHist, surfImage, m_xPars);
					}
					else if (m_xPars.iSingleChannel < TPixelType::ChannelCount)
					{
						switch (m_xPars.iSingleChannel)
						{
						case 0:
							Kernel::ProcessChannel<TPixelType, TDataType, 0>
								CLU_KERNEL_CONFIG()
								(m_deviHist, surfImage, m_xPars);
							break;

						case 1:
							Kernel::ProcessChannel<TPixelType, TDataType, 1>
								CLU_KERNEL_CONFIG()
								(m_deviHist, surfImage, m_xPars);
							break;

						case 2:
							Kernel::ProcessChannel<TPixelType, TDataType, 2>
								CLU_KERNEL_CONFIG()
								(m_deviHist, surfImage, m_xPars);
							break;

						case 3:
							Kernel::ProcessChannel<TPixelType, TDataType, 3>
								CLU_KERNEL_CONFIG()
								(m_deviHist, surfImage, m_xPars);
							break;
						}
					}
					else
					{
						throw CLU_EXCEPTION("Invalid channel");
					}
				}


				template<typename TPixelType>
				void CDriver::_SelectDataType(const Clu::Cuda::_CDeviceSurface& surfImage)
				{
					// Store the number of channels processed.
					m_uChannelCount = TPixelType::ChannelCount;

					switch (surfImage.Format().eDataType)
					{
					case EDataType::UInt8:
						_SelectAlgo<TPixelType, T_UInt8>(surfImage);
						break;

					case EDataType::UInt16:
						_SelectAlgo<TPixelType, T_UInt16>(surfImage);
						break;

					case EDataType::UInt32:
						_SelectAlgo<TPixelType, T_UInt32>(surfImage);
						break;

					case EDataType::Int8:
						_SelectAlgo<TPixelType, T_Int8>(surfImage);
						break;

					case EDataType::Int16:
						_SelectAlgo<TPixelType, T_Int16>(surfImage);
						break;

					case EDataType::Int32:
						_SelectAlgo<TPixelType, T_Int32>(surfImage);
						break;

					case EDataType::Single:
						_SelectAlgo<TPixelType, T_Single>(surfImage);
						break;

					default:
						throw CLU_EXCEPTION("Unsupported image data type");
					}
				}

				void CDriver::_SelectPixelType(const Clu::Cuda::_CDeviceSurface& surfImage)
				{
					switch (surfImage.Format().ePixelType)
					{
					case EPixelType::Lum:
						_SelectDataType<T_Lum>(surfImage);
						break;

					case EPixelType::LumA:
						_SelectDataType<T_LumA>(surfImage);
						break;

					case EPixelType::RGBA:
						_SelectDataType<T_RGBA>(surfImage);
						break;

					case EPixelType::BGRA:
						_SelectDataType<T_BGRA>(surfImage);
						break;

					default:
						throw CLU_EXCEPTION("Unsupported image pixel type");
					}
				}



				double CDriver::Run(const Clu::Cuda::_CDeviceSurface& surfImage)
				{
					memset(m_imgHist.DataPointer(), 0, m_imgHist.ByteCount());
					m_deviHist.CopyFrom(m_imgHist);

					try
					{
						SafeBegin(KernelName());
						try
						{
							_SelectPixelType(surfImage);
						}
						CLU_CATCH_RETHROW_ALL(CLU_S "Error while running kernel '" << KernelName() << "'");
						SafeEnd(KernelName());
						LogLastProcessTime(KernelName());
					}
					CLU_CATCH_RETHROW_ALL(CLU_S "Error processing kernel '" << KernelName() << "'");

					m_deviHist.CopyInto(m_imgHist);
					m_pixTotalCount.SetZero();

					TPixelHist *pHist = (TPixelHist*)m_imgHist.DataPointer();

					for (unsigned uBucketIdx = 0; uBucketIdx < m_xPars.uBucketCount; ++uBucketIdx, ++pHist)
					{
						for (unsigned uChannel = 0; uChannel < m_uChannelCount; ++uChannel)
						{
							m_pixTotalCount[uChannel] += (*pHist)[uChannel];
						}
					}

					return LastProcessTime();
				}



				void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat, const SParameter& xPars)
				{
					m_xPars = xPars;

					m_uChannelCount = 0;
					m_pixTotalCount.SetZero();

					Clu::SImageFormat xF(m_xPars.uBucketCount, 1, Clu::SImageType(TPixelHist::PixelTypeId, TPixelHist::DataTypeId));
					m_imgHist.Create(xF);
					m_deviHist.Create(xF);


					EvalThreadConfigBlockSize(xDevice, xFormat
						, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
						, 0, 0, 0, 0 // Offsets
						, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
						, NumberOfRegisters
						, false // Use also partial blocks
						);
				}

				double CDriver::_GetValueAtPercentCount(double dPercent, double dValueMin, double dValueMax, double dTotalCount, unsigned uChannel)
				{
					unsigned uCount = 0;
					unsigned uBucketIdx = 0;
					TPixelHist *pHist = (TPixelHist*)m_imgHist.DataPointer();

					for (uBucketIdx = 0; uBucketIdx < m_xPars.uBucketCount; ++uBucketIdx, ++pHist)
					{
						uCount += (*pHist)[uChannel];
						if (double(uCount) / dTotalCount >= dPercent)
						{
							break;
						}
					}

					double dValue = double(uBucketIdx) / double(m_xPars.uBucketCount - 1) * (dValueMax - dValueMin) + dValueMin;

					return dValue;
				}

				void CDriver::_GetValueAtPercentCount(double& dResultMin, double& dResultMax, double dPercentMin, double dPercentMax
					, double dValueMin, double dValueMax, double dTotalCount, unsigned uChannel)
				{
					unsigned uCount = 0;
					unsigned uBucketIdx = 0;
					TPixelHist *pHist = (TPixelHist*)m_imgHist.DataPointer();

					for (uBucketIdx = 0; uBucketIdx < m_xPars.uBucketCount; ++uBucketIdx, ++pHist)
					{
						uCount += (*pHist)[uChannel];
						if (double(uCount) / dTotalCount >= dPercentMin)
						{
							break;
						}
					}

					dResultMin = double(uBucketIdx) / double(m_xPars.uBucketCount - 1) * (dValueMax - dValueMin) + dValueMin;
					
					if (uBucketIdx + 1 >= m_xPars.uBucketCount)
					{
						dResultMax = dResultMin;
					}
					else
					{
						++uBucketIdx;
						++pHist;
						for (; uBucketIdx < m_xPars.uBucketCount; ++uBucketIdx, ++pHist)
						{
							uCount += (*pHist)[uChannel];
							if (double(uCount) / dTotalCount >= dPercentMax)
							{
								break;
							}
						}
					
						dResultMax = double(uBucketIdx) / double(m_xPars.uBucketCount - 1) * (dValueMax - dValueMin) + dValueMin;
					}
				}


				template<typename TArray>
				void CDriver::GetData(TArray& aData, unsigned uChannel)
				{
					using TValue = typename TArray::TValue;
					TPixelHist *pDataSrc = (TPixelHist*) m_imgHist.DataPointer();
					int iCount = m_imgHist.Format().iWidth;

					aData.Resize(iCount);
					TValue *pDataTrg = (TValue *) aData.DataPointer();

					for (int i = 0; i < iCount; ++i, ++pDataSrc, ++pDataTrg)
					{
						*pDataTrg = (TValue)((*pDataSrc)[uChannel]);
					}
				}

				template void CDriver::GetData(Clu::CIArrayInt32& aData, unsigned uChannel);
				template void CDriver::GetData(Clu::CIArrayInt64& aData, unsigned uChannel);


				TPixelValueRange CDriver::GetValueAtPercentCount(double dPercent)
				{
					using TFloat = TPixelValueRange::TData;
					TPixelValueRange pixResult;

					pixResult.SetZero();

					if (m_xPars.iSingleChannel < 0)
					{
						for (unsigned uChannel = 0; uChannel < m_uChannelCount; ++uChannel)
						{
							pixResult[uChannel]
								= _GetValueAtPercentCount(dPercent
									, double(m_xPars.pixMin[uChannel]), double(m_xPars.pixMax[uChannel])
									, double(m_pixTotalCount[uChannel]), uChannel);
						}
					}
					else
					{
						unsigned uChannel = (unsigned)m_xPars.iSingleChannel;
						if (uChannel >= m_uChannelCount)
						{
							throw CLU_EXCEPTION("Invalid single channel");
						}

						pixResult[uChannel] = _GetValueAtPercentCount(dPercent
							, double(m_xPars.pixMin[uChannel]), double(m_xPars.pixMax[uChannel])
							, double(m_pixTotalCount[uChannel]), uChannel);
					}

					return pixResult;
				}

				double CDriver::GetValueAtPercentCount(double dPercent, unsigned uChannel)
				{
					try
					{
						using TFloat = TPixelValueRange::TData;

						if ( (uChannel >= m_uChannelCount)
							|| (m_xPars.iSingleChannel >= 0 && uChannel != unsigned(m_xPars.iSingleChannel)))
						{
							throw CLU_EXCEPTION("Invalid channel");
						}

						return _GetValueAtPercentCount(dPercent
									, double(m_xPars.pixMin[uChannel]), double(m_xPars.pixMax[uChannel])
									, double(m_pixTotalCount[uChannel]), uChannel);
					}
					CLU_CATCH_RETHROW_ALL("Error obtaining value at percentage count")
				}

				void CDriver::GetValueAtPercentCount(double & dValueMin, double & dValueMax, double dPercentMin, double dPercentMax, unsigned uChannel)
				{
					try
					{
						using TFloat = TPixelValueRange::TData;

						if ((uChannel >= m_uChannelCount)
							|| (m_xPars.iSingleChannel >= 0 && uChannel != unsigned(m_xPars.iSingleChannel)))
						{
							throw CLU_EXCEPTION("Invalid channel");
						}

						_GetValueAtPercentCount(dValueMin, dValueMax, dPercentMin, dPercentMax
							, double(m_xPars.pixMin[uChannel]), double(m_xPars.pixMax[uChannel])
							, double(m_pixTotalCount[uChannel]), uChannel);
					}
					CLU_CATCH_RETHROW_ALL("Error obtaining value at percentage count")
				}

			} // Histogram
		} // Statistics
	} // Cuda
} // Clu


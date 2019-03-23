////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Filter.Sobel.cu
//
// summary:   filter. sobel class
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

#include "Filter.Sobel.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

#define CLU_DEBUG_KERNEL
#include "CluTec.Cuda.Base/Kernel.Debug.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Filter
		{
			namespace Sobel
			{
				namespace Kernel
				{
					using namespace Clu::Cuda::Kernel;

					struct SConfig
					{
						static const int WarpsPerBlockX = 4;
						static const int WarpsPerBlockY = 1;
						static const int ThreadCountX = 8;
						static const int ThreadCountY = 16;
						static const int BlockSizeX = ThreadCountX;
						static const int BlockSizeY = ThreadCountY;
					};


					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	The algorithm parameters. </summary>
					///
					/// <value>	. </value>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					__constant__ _SParameter c_xPars;

					////////////////////////////////////////////////////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TConfig>
					__device__ __forceinline__ void GetSrcPos(int& iSrcX, int& iSrcY)
					{
						iSrcX = int(blockIdx.x * TConfig::BlockSizeX + threadIdx.x % TConfig::BlockSizeX);
						iSrcY = int(blockIdx.y * TConfig::BlockSizeY + threadIdx.x / TConfig::BlockSizeX);
					}


					template<int t_iRadius>
					struct SOpDerivative
					{};

					template<>
					struct SOpDerivative<1>
					{
						static const int Radius = 1;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 1;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return -1;
							if (iIdx == 1) return 0;
							if (iIdx == 2) return 1;
							return 0;
						}

					};

					template<>
					struct SOpDerivative<2>
					{
						static const int Radius = 2;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 3;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return -1;
							if (iIdx == 1) return -2;
							if (iIdx == 2) return 0;
							if (iIdx == 3) return 2;
							if (iIdx == 4) return 1;
							return 0;
						}

					};

					template<>
					struct SOpDerivative<3>
					{
						static const int Radius = 3;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 10;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return -1;
							if (iIdx == 1) return -4;
							if (iIdx == 2) return -5;
							if (iIdx == 3) return 0;
							if (iIdx == 4) return 5;
							if (iIdx == 5) return 4;
							if (iIdx == 6) return 1;
							return 0;
						}

					};

					template<>
					struct SOpDerivative<4>
					{
						static const int Radius = 4;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 35;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return -1;
							if (iIdx == 1) return -6;
							if (iIdx == 2) return -14;
							if (iIdx == 3) return -14;
							if (iIdx == 4) return 0;
							if (iIdx == 5) return 14;
							if (iIdx == 6) return 14;
							if (iIdx == 7) return 6;
							if (iIdx == 8) return 1;
							return 0;
						}

					};

					template<int t_iRadius>
					struct SOpBinomial
					{};

					template<>
					struct SOpBinomial<1>
					{
						static const int Radius = 1;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 4;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return 1;
							if (iIdx == 1) return 2;
							if (iIdx == 2) return 1;
							return 0;
						}
					};

					template<>
					struct SOpBinomial<2>
					{
						static const int Radius = 2;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 16;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return 1;
							if (iIdx == 1) return 4;
							if (iIdx == 2) return 6;
							if (iIdx == 3) return 4;
							if (iIdx == 4) return 1;
							return 0;
						}
					};

					template<>
					struct SOpBinomial<3>
					{
						static const int Radius = 3;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 64;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return 1;
							if (iIdx == 1) return 6;
							if (iIdx == 2) return 15;
							if (iIdx == 3) return 20;
							if (iIdx == 4) return 15;
							if (iIdx == 5) return 6;
							if (iIdx == 6) return 1;
							return 0;
						}
					};

					template<>
					struct SOpBinomial<4>
					{
						static const int Radius = 4;
						static const int Size = 2 * Radius + 1;

						static const int Divider = 256;

						__device__ __forceinline__ static int Mask(const int iIdx)
						{
							if (iIdx == 0) return 1;
							if (iIdx == 1) return 8;
							if (iIdx == 2) return 28;
							if (iIdx == 3) return 56;
							if (iIdx == 4) return 70;
							if (iIdx == 5) return 56;
							if (iIdx == 6) return 28;
							if (iIdx == 7) return 8;
							if (iIdx == 8) return 1;
							return 0;
						}
					};

					template<typename TPixelSrc, typename TOp, int t_iDoNorm>
					__device__ __forceinline__ int DoConvolveH(const Clu::Cuda::_CDeviceSurface& xImageSrc, int iSrcX, int iSrcY)
					{
						int iValue = 0;
						iSrcX -= TOp::Radius;
						for (int iIdxX = 0; iIdxX < TOp::Size; ++iIdxX)
						{

							const int iSrcVal = int(xImageSrc.ReadPixel2D<TPixelSrc>(iSrcX, iSrcY).r());
							const int iMaskVal = TOp::Mask(iIdxX);
							iValue += iMaskVal * iSrcVal;

							//if (Debug::IsBlock(10, 10))
							//{
							//	printf("%d, %d, %d: %d, %d -> %d\n", threadIdx.x, threadIdx.y, iSrcX, iMaskVal, iSrcVal, iValue);
							//}

							++iSrcX;
						}

						iValue /= (t_iDoNorm * TOp::Divider + (1 - t_iDoNorm));
						return iValue;
					}

					template<typename TPixelSrc, typename TOp, int t_iDoNorm>
					__device__ __forceinline__ int DoConvolveV(const Clu::Cuda::_CDeviceSurface& xImageSrc, int iSrcX, int iSrcY)
					{
						int iValue = 0;
						iSrcY -= TOp::Radius;
						for (int iIdxY = 0; iIdxY < TOp::Size; ++iIdxY)
						{
							iValue += TOp::Mask(iIdxY) * int(xImageSrc.ReadPixel2D<TPixelSrc>(iSrcX, iSrcY).r());
							++iSrcY;
						}

						iValue /= (t_iDoNorm * TOp::Divider + (1 - t_iDoNorm));
						return iValue;
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Horizontal convolution. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="t_iRadius">	Type of the radius. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TPixelTrg, typename TPixelSrc, typename TOp, int t_iDoNorm>
					__global__ void ConvolveH(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
					{
						using TDataTrg = typename TPixelTrg::TData;

						int iSrcX, iSrcY;

						GetSrcPos<SConfig>(iSrcX, iSrcY);

						if (!xImageSrc.IsInside(iSrcX, iSrcY, TOp::Radius, 0))
						{
							return;
						}

						int iValue = DoConvolveH<TPixelSrc, TOp, t_iDoNorm>(xImageSrc, iSrcX, iSrcY);

						xImageTrg.Write2D<TDataTrg>(TDataTrg(iValue), iSrcX, iSrcY);
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Vertical convolution. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel trg. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="t_iRadius">	Type of the radius. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TPixelTrg, typename TPixelSrc, typename TOp, int t_iDoNorm>
					__global__ void ConvolveV(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
					{
						using TDataTrg = typename TPixelTrg::TData;

						int iSrcX, iSrcY;

						GetSrcPos<SConfig>(iSrcX, iSrcY);

						if (!xImageSrc.IsInside(iSrcX, iSrcY, 0, TOp::Radius))
						{
							return;
						}

						int iValue = DoConvolveV<TPixelSrc, TOp, t_iDoNorm>(xImageSrc, iSrcX, iSrcY);

						xImageTrg.Write2D<TDataTrg>(TDataTrg(iValue), iSrcX, iSrcY);
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Abs graduated. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel trg. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="TOp">			Type of the operation. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TPixelTrg, typename TPixelSrc, int t_iRadius>
					__global__ void AbsGrad(Clu::Cuda::_CDeviceSurface imgTrg
						, Clu::Cuda::_CDeviceSurface imgDerivH
						, Clu::Cuda::_CDeviceSurface imgDerivV)
					{

						using TDataTrg = typename TPixelTrg::TData;
						static const int c_iRadius = t_iRadius;

						int iSrcX, iSrcY;

						GetSrcPos<SConfig>(iSrcX, iSrcY);

						if (!imgTrg.IsInside(iSrcX, iSrcY, c_iRadius, c_iRadius))
						{
							return;
						}

						float fValueH = (float)DoConvolveV<TPixelSrc, SOpBinomial<c_iRadius>, 0>(imgDerivH, iSrcX, iSrcY);
						float fValueV = (float)DoConvolveH<TPixelSrc, SOpBinomial<c_iRadius>, 0>(imgDerivV, iSrcX, iSrcY);

						//if (Debug::IsBlock(10, 10))
						//{
						//	printf("%d, %d: %g, %g\n", threadIdx.x, threadIdx.y, fValueH, fValueV);
						//}

						fValueH /= float(SOpDerivative<c_iRadius>::Divider) * float(SOpBinomial<c_iRadius>::Divider);
						fValueV /= float(SOpDerivative<c_iRadius>::Divider) * float(SOpBinomial<c_iRadius>::Divider);
						float fGrad = sqrtf(fValueH * fValueH + fValueV * fValueV);

						// Gamma adjustment of resultant image
						fGrad = c_xPars.fScale 
								* powf(fGrad / float(Clu::NumericLimits<TDataTrg>::Max()), c_xPars.fGamma) 
								* float(Clu::NumericLimits<TDataTrg>::Max());

						int iGrad = int(floor(fGrad + 0.5f));
						iGrad = min(max(iGrad, 0), Clu::NumericLimits<TDataTrg>::Max());

						//if (Debug::IsBlock(20, 20) && fGrad < 0.0f)
						//{
						//	printf("%d, %d: %g, %g, %g, %d\n", threadIdx.x, threadIdx.y, fValueH, fValueV, fGrad, iGrad);
						//}

						imgTrg.Write2D<TDataTrg>(TDataTrg(iGrad), iSrcX, iSrcY);
					}


				} // namespace Kernel

				// /////////////////////////////////////////////////////////////////////////////////////////////////////////
				// /////////////////////////////////////////////////////////////////////////////////////////////////////////

				template<EConfig t_eConfig>
				void CDriver::_DoConfigure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
				{
					using Config = SConfig<t_eConfig>;

					unsigned nOffsetLeft = Config::Radius;
					unsigned nOffsetRight = Config::Radius;
					unsigned nOffsetTop = 0; 
					unsigned nOffsetBottom = 0;

					EvalThreadConfigBlockSize(xDevice, xFormat
						, Config::BlockSizeX, Config::BlockSizeY
						, nOffsetLeft, nOffsetRight, nOffsetTop, nOffsetBottom
						, Config::WarpsPerBlockX, Config::WarpsPerBlockY
						, Config::NumberOfRegisters
						, false // do process partial blocks
						, KID_ConvolveH // Kernel Id
					);

					nOffsetLeft = 0;
					nOffsetRight = 0;
					nOffsetTop = Config::Radius;
					nOffsetBottom = Config::Radius;

					EvalThreadConfigBlockSize(xDevice, xFormat
						, Config::BlockSizeX, Config::BlockSizeY
						, nOffsetLeft, nOffsetRight, nOffsetTop, nOffsetBottom
						, Config::WarpsPerBlockX, Config::WarpsPerBlockY
						, Config::NumberOfRegisters
						, false // do process partial blocks
						, KID_ConvolveV // Kernel Id
					);

					nOffsetLeft = Config::Radius;
					nOffsetRight = Config::Radius;
					nOffsetTop = Config::Radius;
					nOffsetBottom = Config::Radius;

					EvalThreadConfigBlockSize(xDevice, xFormat
						, Config::BlockSizeX, Config::BlockSizeY
						, nOffsetLeft, nOffsetRight, nOffsetTop, nOffsetBottom
						, Config::WarpsPerBlockX, Config::WarpsPerBlockY
						, Config::NumberOfRegisters
						, false // do process partial blocks
						, KID_AbsGrad // Kernel Id
					);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Configures. </summary>
				///
				/// <param name="xConfig">	[in,out] The configuration. </param>
				/// <param name="xDevice">	The device. </param>
				/// <param name="xFormat">	Describes the format to use. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

#define _CLU_DO_CONFIG(theId) \
			case theId: \
				_DoConfigure<theId>(xDevice, xFormat); \
				break

				void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat,
					const SParameter& xPars)
				{
					m_xPars = xPars;

					switch (m_xPars.eConfig)
					{
						_CLU_DO_CONFIG(EConfig::Patch_3x3);
						_CLU_DO_CONFIG(EConfig::Patch_5x5);
						_CLU_DO_CONFIG(EConfig::Patch_7x7);
						_CLU_DO_CONFIG(EConfig::Patch_9x9);

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}

				}
#undef _CLU_DO_CONFIG

				// /////////////////////////////////////////////////////////////////////////////////////////////////////////
				// _DoProcess
				// /////////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelSrc, typename TPixelSum, typename TConfig>
				void CDriver::_DoProcess(Clu::Cuda::_CDeviceSurface& xImageTrg
					, const Clu::Cuda::_CDeviceSurface& xImageSrc
					, Clu::Cuda::CDeviceSurface& xImageTempH
					, Clu::Cuda::CDeviceSurface& xImageTempV)
				{
					static const int Radius = TConfig::Radius;

					// Horizontal derivative
					Kernel::ConvolveH<TPixelSum, TPixelSrc, Kernel::SOpDerivative<Radius>, 0 >
						CLU_KERNEL_CONFIG_(KID_ConvolveH)
						(xImageTempH, xImageSrc);

					// Vertical derivative
					Kernel::ConvolveV<TPixelSum, TPixelSrc, Kernel::SOpDerivative<Radius>, 0 >
						CLU_KERNEL_CONFIG_(KID_ConvolveV)
						(xImageTempV, xImageSrc);

					// Absolute Gradient
					Kernel::AbsGrad<TPixelSrc, TPixelSum, Radius>
						CLU_KERNEL_CONFIG_(KID_AbsGrad)
						(xImageTrg, xImageTempH, xImageTempV);
				}

				// /////////////////////////////////////////////////////////////////////////////////////////////////////////
				// _SelectConfig
				// /////////////////////////////////////////////////////////////////////////////////////////////////////////

#define _CLU_DO_PROCESS(theId) \
			case theId: \
				_DoProcess<TPixelSrc, TPixelSum, SConfig<theId> > \
					(xImageTrg, xImageSrc, xImageTempH, xImageTempV); \
				break

				template<typename TPixelSrc, typename TPixelSum>
				void CDriver::_SelectConfig(Clu::Cuda::_CDeviceSurface& xImageTrg
					, const Clu::Cuda::_CDeviceSurface& xImageSrc
					, Clu::Cuda::CDeviceSurface& xImageTempH
					, Clu::Cuda::CDeviceSurface& xImageTempV)
				{
					Clu::SImageFormat xF = xImageSrc.Format();
					xF.eDataType = TPixelSum::DataTypeId;
					xF.ePixelType = TPixelSum::PixelTypeId;

					xImageTempH.Create(xF);
					xImageTempV.Create(xF);

					switch (m_xPars.eConfig)
					{
						_CLU_DO_PROCESS(EConfig::Patch_3x3);
						_CLU_DO_PROCESS(EConfig::Patch_5x5);
						_CLU_DO_PROCESS(EConfig::Patch_7x7);
						_CLU_DO_PROCESS(EConfig::Patch_9x9);

					default:
						throw CLU_EXCEPTION("Invalid algorithm configuration.");
					}

				}
#undef _CLU_DO_PROCESS

				// /////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Process
				// /////////////////////////////////////////////////////////////////////////////////////////////////////////

				void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc
					, Clu::Cuda::CDeviceSurface& xImageTempH
					, Clu::Cuda::CDeviceSurface& xImageTempV)
				{
					if (!xImageTrg.IsEqualFormat(xImageSrc.Format()))
					{
						throw CLU_EXCEPTION("Input and output images do not have the same format");
					}

					Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

					if (xImageSrc.IsOfType<Clu::TPixel_Lum_UInt8>()
						&& xImageTrg.IsOfType<Clu::TPixel_Lum_UInt8>())
					{
						_SelectConfig<Clu::TPixel_Lum_UInt8, Clu::TPixel_Lum_Int16>(xImageTrg, xImageSrc, xImageTempH, xImageTempV);
					}
					else
					{
						throw CLU_EXCEPTION("Pixel types of given images not supported");
					}

				}




			} // Sobel
		} // Filter
	} // Cuda
} // Clu


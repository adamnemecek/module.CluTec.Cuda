////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      ColorDisparityImage.cu
//
// summary:   color disparity image class
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

#include "ColorDisparityImage.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"
#include "CluTec.Cuda.Base/PixelTypeInfo.h"
#include "CluTec.Cuda.Base/Conversion.h"


namespace Clu
{
	namespace Cuda
	{
		namespace ColorDisparityImage
		{
			namespace Kernel
			{
				__constant__ Clu::Cuda::ColorDisparityImage::_SParameter c_xPars;


				template<EStyle t_eStyle>
				struct SAlgo
				{};


				template<>
				struct SAlgo<EStyle::Analysis>
				{
					template<typename TPixelDisp>
					__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceSurface xGrayImage, Clu::Cuda::_CDeviceSurface xDisparityImage, TPixelDisp)
					{
						using TColorComp = typename TPixelColor::TData;

						using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
						using TDisp = typename TPixelDisp::TData;

						const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

						if (!xDisparityImage.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						const float fDispRange = float(c_xPars.uDispMax - c_xPars.uDispMin);

						TPixelColor pixGray;
						TDispTuple pixDisp = xDisparityImage.Read2D<TDispTuple>(nSrcX, nSrcY);

						if (pixDisp.x >= TDisp(EDisparityId::First) && pixDisp.x <= TDisp(EDisparityId::Last))
						{
							float fDisp = float(pixDisp.x - TDisp(c_xPars.uDispMin)) / fDispRange;
							TColorComp xValue = Clu::NormFloatTo<TColorComp>(fDisp);

							pixGray = Clu::Cuda::Make<TPixelColor>(xValue, xValue, xValue, Clu::NumericLimits<TColorComp>::Max());
						}
						else if (pixDisp.x == TDisp(EDisparityId::Unknown))
						{
							pixGray = c_xPars.pixUnknown;
						}
						else if (pixDisp.x == TDisp(EDisparityId::CannotEvaluate))
						{
							pixGray = c_xPars.pixCannotEvaluate;
						}
						else if (pixDisp.x == TDisp(EDisparityId::Saturated))
						{
							pixGray = c_xPars.pixSaturated;
						}
						else if (pixDisp.x == TDisp(EDisparityId::NotSpecific))
						{
							pixGray = c_xPars.pixNotSpecific;
						}
						else if (pixDisp.x == TDisp(EDisparityId::NotFound))
						{
							pixGray = c_xPars.pixNotFound;
						}
						else if (pixDisp.x == TDisp(EDisparityId::NotUnique))
						{
							pixGray = c_xPars.pixNotUnique;
						}
						else if (pixDisp.x == TDisp(EDisparityId::Inconsistent))
						{
							pixGray = c_xPars.pixInconsistent;
						}
						else
						{
							pixGray = Clu::Cuda::Make<TPixelColor>(255, 0, 0, 255);
						}

						xGrayImage.WritePixel2D<TPixelColor>(pixGray, nSrcX, nSrcY);
					}

				};

				template<>
				struct SAlgo<EStyle::Gray>
				{
					template<typename TPixelDisp>
					__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceSurface xGrayImage, Clu::Cuda::_CDeviceSurface xDisparityImage, TPixelDisp)
					{
						using TGrayComp = typename TPixelGray::TData;

						using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
						using TDisp = typename TPixelDisp::TData;

						const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

						if (!xDisparityImage.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						const float fDispRange = float(c_xPars.uDispMax - c_xPars.uDispMin);

						TPixelGray pixGray;
						TDispTuple pixDisp = xDisparityImage.Read2D<TDispTuple>(nSrcX, nSrcY);

						if (pixDisp.x >= TDisp(EDisparityId::First) && pixDisp.x <= TDisp(EDisparityId::Last))
						{
							float fDisp = float(pixDisp.x - TDisp(c_xPars.uDispMin)) / fDispRange;
							pixGray.r() = Clu::NormFloatTo<TGrayComp>(fDisp);

						}
						else
						{
							pixGray.r() = TGrayComp(0);
						}

						xGrayImage.WritePixel2D<TPixelGray>(pixGray, nSrcX, nSrcY);
					}
				};

				template<>
				struct SAlgo<EStyle::Color>
				{
					template<typename TPixelDisp>
					__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceSurface xColorImage, Clu::Cuda::_CDeviceSurface xDisparityImage, TPixelDisp)
					{
						using TColorComp = typename TPixelColor::TData;

						using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
						using TDisp = typename TPixelDisp::TData;

						const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

						if (!xDisparityImage.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						const float fDispRange = float(c_xPars.uDispMax - c_xPars.uDispMin);

						TPixelColor pixColor;
						TDispTuple pixDisp = xDisparityImage.Read2D<TDispTuple>(nSrcX, nSrcY);

						if (pixDisp.x >= TDisp(EDisparityId::First) && pixDisp.x <= TDisp(EDisparityId::Last))
						{
							float fDisp = float(pixDisp.x - TDisp(c_xPars.uDispMin)) / fDispRange;
							pixColor = Clu::Cuda::NormFloatToColor<TPixelColor>(fDisp);

						}
						else
						{
							pixColor = Clu::Cuda::Make<TPixelColor>(0, 0, 0, 255);
						}


						xColorImage.WritePixel2D<TPixelColor>(pixColor, nSrcX, nSrcY);
					}
				};


				template<>
				struct SAlgo<EStyle::DispAbs>
				{
					template<typename TPixelDisp>
					__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceSurface xGrayImage, Clu::Cuda::_CDeviceSurface xDisparityImage, TPixelDisp)
					{
						using TGrayComp = typename TPixelGray::TData;

						using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
						using TDisp = typename TPixelDisp::TData;

						const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

						if (!xDisparityImage.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelGray pixGray;
						TDispTuple pixDisp = xDisparityImage.Read2D<TDispTuple>(nSrcX, nSrcY);

						if (pixDisp.x >= TDisp(EDisparityId::First) && pixDisp.x <= TDisp(EDisparityId::Last))
						{
							pixGray.r() = pixDisp.x;
						}
						else
						{
							pixGray.r() = TGrayComp(0);
						}

						xGrayImage.WritePixel2D<TPixelGray>(pixGray, nSrcX, nSrcY);
					}
				};

				template<>
				struct SAlgo<EStyle::DispRel>
				{
					template<typename TPixelDisp>
					__device__ __forceinline__ static void Process(Clu::Cuda::_CDeviceSurface xGrayImage, Clu::Cuda::_CDeviceSurface xDisparityImage, TPixelDisp)
					{
						using TGrayComp = typename TPixelGray::TData;

						using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
						using TDisp = typename TPixelDisp::TData;

						const int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						const int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);

						if (!xDisparityImage.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelGray pixGray;
						TDispTuple pixDisp = xDisparityImage.Read2D<TDispTuple>(nSrcX, nSrcY);

						if (pixDisp.x >= TDisp(EDisparityId::First) && pixDisp.x <= TDisp(EDisparityId::Last))
						{
							pixGray.r() = (TDisp(0x7FFF) + c_xPars.uDispInfOffset) - (pixDisp.x - TDisp(EDisparityId::First));
						}
						else
						{
							pixGray.r() = TGrayComp(0);
						}

						xGrayImage.WritePixel2D<TPixelGray>(pixGray, nSrcX, nSrcY);
					}
				};

				template<typename TPixelDisp, EStyle t_eStyle>
				__global__ void Process(Clu::Cuda::_CDeviceSurface xColorImage, Clu::Cuda::_CDeviceSurface xDisparityImage)
				{
					SAlgo<t_eStyle>::Process(xColorImage, xDisparityImage, TPixelDisp());
				}

			} // namespace Kernel

			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// ///////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// DRIVER
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////
			// //////////////////////////////////////////////////////////////////////////////////////////////////////////

			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
				, const SParameter& xPars)
			{
				m_xPars = xPars;
				EvalThreadConfig(xDevice, xFormat, 0, 0, NumberOfRegisters);
			}

			template<typename TPixelDisp>
			void CDriver::SelectStyle(Clu::Cuda::_CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle)
			{
				switch (eStyle)
				{
				case Clu::Cuda::ColorDisparityImage::EStyle::Analysis:
					Kernel::Process<TPixelDisp, EStyle::Analysis>
						CLU_KERNEL_CONFIG()
						(xColorImage, xDisparityImage);
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::Gray:
					Kernel::Process<TPixelDisp, EStyle::Gray>
						CLU_KERNEL_CONFIG()
						(xColorImage, xDisparityImage);
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::Color:
					Kernel::Process<TPixelDisp, EStyle::Color>
						CLU_KERNEL_CONFIG()
						(xColorImage, xDisparityImage);
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::DispAbs:
					Kernel::Process<TPixelDisp, EStyle::DispAbs>
						CLU_KERNEL_CONFIG()
						(xColorImage, xDisparityImage);
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::DispRel:
					Kernel::Process<TPixelDisp, EStyle::DispRel>
						CLU_KERNEL_CONFIG()
						(xColorImage, xDisparityImage);
					break;

				default:
					throw CLU_EXCEPTION("Given coloring style not supported");
				}
			}

			void CDriver::Process(Clu::Cuda::CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle)
			{
				SImageFormat xF(xDisparityImage.Format());

				switch (eStyle)
				{
				case Clu::Cuda::ColorDisparityImage::EStyle::Analysis:
				case Clu::Cuda::ColorDisparityImage::EStyle::Color:
					xF.ePixelType = TPixelColor::PixelTypeId;
					xF.eDataType = TPixelColor::DataTypeId;
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::Gray:
				case Clu::Cuda::ColorDisparityImage::EStyle::DispAbs:
				case Clu::Cuda::ColorDisparityImage::EStyle::DispRel:
					xF.ePixelType = TPixelGray::PixelTypeId;
					xF.eDataType = TPixelGray::DataTypeId;
					break;

				default:
					throw CLU_EXCEPTION("Unsupported image style");
				}

				xColorImage.Create(xF);

				Process((Clu::Cuda::_CDeviceSurface&)xColorImage, xDisparityImage, eStyle);
			}

			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xColorImage, const Clu::Cuda::_CDeviceSurface& xDisparityImage, EStyle eStyle)
			{
				switch (eStyle)
				{
				case Clu::Cuda::ColorDisparityImage::EStyle::Analysis:
				case Clu::Cuda::ColorDisparityImage::EStyle::Color:
					if (!xColorImage.IsOfType<TPixelColor>())
					{
						throw CLU_EXCEPTION("Given color image is not of the correct type");
					}
					break;

				case Clu::Cuda::ColorDisparityImage::EStyle::Gray:
				case Clu::Cuda::ColorDisparityImage::EStyle::DispAbs:
				case Clu::Cuda::ColorDisparityImage::EStyle::DispRel:
					if (!xColorImage.IsOfType<TPixelGray>())
					{
						throw CLU_EXCEPTION("Given color image is not of the correct type");
					}
					break;

				default:
					throw CLU_EXCEPTION("Unsupported image style");
				}

				if (!xColorImage.IsEqualSize(xDisparityImage.Format()))
				{
					throw CLU_EXCEPTION("Color and disparity images are not of the same size");
				}

				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

				if (xDisparityImage.IsOfType<TPixelDispEx>())
				{
					SelectStyle<TPixelDispEx>(xColorImage, xDisparityImage, eStyle);
				}
				else if (xDisparityImage.IsOfType<TPixelDisp>())
				{
					SelectStyle<TPixelDisp>(xColorImage, xDisparityImage, eStyle);
				}
				else if (xDisparityImage.IsOfType<TPixelDispF>())
				{
					SelectStyle<TPixelDispF>(xColorImage, xDisparityImage, eStyle);
				}
				else
				{
					throw CLU_EXCEPTION("Disparity image pixel type not supported");
				}

			}


		} // ImgProc
	} // Cuda
} // Clu


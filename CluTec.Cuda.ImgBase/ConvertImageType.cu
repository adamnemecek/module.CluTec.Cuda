////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      ConvertImageType.cu
//
// summary:   convert image type class
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

#include "ConvertImageType.h"

#include "CluTec.Types1/Pixel.h"

#include "CluTec.Math/Conversion.h"
#include "CluTec.Math/MapPixelValue.h"
#include "CluTec.Math/Static.Vector.h"
#include "CluTec.Math/Static.Matrix.h"

#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceLayerImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"


namespace Clu
{
	namespace Cuda
	{
		namespace ConvertImageType
		{
			enum class EConvertType
			{
				ColToCol = 0,
				LumToLum,
				ColToLum,
				LumToCol,
			};

			namespace Kernel
			{
				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>
				/// The general declaration for a struct that contains a conversion device function for the given
				/// conversion type. Using a partially specialized struct templates allows for the translation of
				/// an enum into different functions at compile time.
				/// </summary>
				///
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				/// <typeparam name="eConvType">		Type of the convert type. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc, EConvertType eConvType>
				struct SConvertAlgo
				{};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Specialization for mapping a color to a color pixel. </summary>
				///
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc>
				struct SConvertAlgo<TPixelTypeTrg, TDataTypeTrg, TPixelTypeSrc, TDataTypeSrc, EConvertType::ColToCol>
				{
					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Col to col. </summary>
					///
					/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
					/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
					/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
					/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
					/// <typeparam name="TImageSrc">		Type of the image source. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TImageTrg, typename TImageSrc>
					__device__ static void Convert(TImageTrg xImageTrg, TImageSrc xImageSrc)
					{
						using TPixelTrg = Clu::SPixel<TPixelTypeTrg, TDataTypeTrg>;
						using TPixelSrc = Clu::SPixel<TPixelTypeSrc, TDataTypeSrc>;
						using TDataTrg = typename TPixelTrg::TData;

						int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);
						if (!xImageTrg.IsInside(nSrcX, nSrcY) || !xImageSrc.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelSrc pixSrc;
						xImageSrc.ReadPixel2D(pixSrc, nSrcX, nSrcY);
						TPixelTrg pixTrg;

						pixTrg.r() = Clu::MapPixelValue<TDataTrg>(pixSrc.r());
						pixTrg.g() = Clu::MapPixelValue<TDataTrg>(pixSrc.g());
						pixTrg.b() = Clu::MapPixelValue<TDataTrg>(pixSrc.b());

						if (TPixelTrg::AlphaCount > 0)
						{
							if (TPixelSrc::AlphaCount > 0)
							{
								pixTrg.a() = Clu::MapPixelValue<TDataTrg>(pixSrc.a());
							}
							else
							{
								pixTrg.a() = Clu::NumericLimits<TDataTrg>::Max();
							}
						}

						xImageTrg.WritePixel2D(pixTrg, nSrcX, nSrcY);
					}
				};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Specialization for mapping a luminance to a luminance pixel. </summary>
				///
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc>
				struct SConvertAlgo<TPixelTypeTrg, TDataTypeTrg, TPixelTypeSrc, TDataTypeSrc, EConvertType::LumToLum>
				{
					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Lum to lum. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel trg. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="TImageSrc">	Type of the image source. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TImageTrg, typename TImageSrc>
					__device__ static void Convert(TImageTrg xImageTrg, TImageSrc xImageSrc)
					{
						using TPixelTrg = Clu::SPixel<TPixelTypeTrg, TDataTypeTrg>;
						using TPixelSrc = Clu::SPixel<TPixelTypeSrc, TDataTypeSrc>;
						using TDataTrg = typename TPixelTrg::TData;

						int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);
						if (!xImageTrg.IsInside(nSrcX, nSrcY) || !xImageSrc.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelSrc pixSrc;
						xImageSrc.ReadPixel2D(pixSrc, nSrcX, nSrcY);
						TPixelTrg pixTrg;

						pixTrg.r() = Clu::MapPixelValue<TDataTrg>(pixSrc.r());

						if (TPixelTrg::AlphaCount > 0)
						{
							if (TPixelSrc::AlphaCount > 0)
							{
								pixTrg.a() = Clu::MapPixelValue<TDataTrg>(pixSrc.a());
							}
							else
							{
								pixTrg.a() = Clu::NumericLimits<TDataTrg>::Max();
							}
						}

						xImageTrg.WritePixel2D(pixTrg, nSrcX, nSrcY);
					}
				};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Specialization for mapping a luminance to a color pixel. </summary>
				///
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc>
				struct SConvertAlgo<TPixelTypeTrg, TDataTypeTrg, TPixelTypeSrc, TDataTypeSrc, EConvertType::LumToCol>
				{
					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Lum to Col. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel trg. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="TImageSrc">	Type of the image source. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TImageTrg, typename TImageSrc>
					__device__ static void Convert(TImageTrg xImageTrg, TImageSrc xImageSrc)
					{
						using TPixelTrg = Clu::SPixel<TPixelTypeTrg, TDataTypeTrg>;
						using TPixelSrc = Clu::SPixel<TPixelTypeSrc, TDataTypeSrc>;
						using TDataTrg = typename TPixelTrg::TData;

						int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);
						if (!xImageTrg.IsInside(nSrcX, nSrcY) || !xImageSrc.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelSrc pixSrc;
						xImageSrc.ReadPixel2D(pixSrc, nSrcX, nSrcY);
						TPixelTrg pixTrg;

						auto xLum = Clu::MapPixelValue<TDataTrg>(pixSrc.r());

						pixTrg.r() = xLum;
						pixTrg.g() = xLum;
						pixTrg.b() = xLum;

						if (TPixelTrg::AlphaCount > 0)
						{
							if (TPixelSrc::AlphaCount > 0)
							{
								pixTrg.a() = Clu::MapPixelValue<TDataTrg>(pixSrc.a());
							}
							else
							{
								pixTrg.a() = Clu::NumericLimits<typename TPixelTrg::TData>::Max();
							}
						}

						xImageTrg.WritePixel2D(pixTrg, nSrcX, nSrcY);
					}
				};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Specialization for mapping a color to a luminance pixel. </summary>
				///
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc>
				struct SConvertAlgo<TPixelTypeTrg, TDataTypeTrg, TPixelTypeSrc, TDataTypeSrc, EConvertType::ColToLum>
				{
					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Color to Luminance. </summary>
					///
					/// <typeparam name="TPixelTrg">	Type of the pixel trg. </typeparam>
					/// <typeparam name="TPixelSrc">	Type of the pixel source. </typeparam>
					/// <typeparam name="TImageSrc">	Type of the image source. </typeparam>
					/// <param name="xImageTrg">	The image trg. </param>
					/// <param name="xImageSrc">	The image source. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TImageTrg, typename TImageSrc>
					__device__ static void Convert(TImageTrg xImageTrg, TImageSrc xImageSrc)
					{
						using TPixelTrg = Clu::SPixel<TPixelTypeTrg, TDataTypeTrg>;
						using TPixelSrc = Clu::SPixel<TPixelTypeSrc, TDataTypeSrc>;
						using TDataTrg = typename TPixelTrg::TData;

						int nSrcX = int(blockIdx.x * blockDim.x + threadIdx.x);
						int nSrcY = int(blockIdx.y * blockDim.y + threadIdx.y);
						if (!xImageTrg.IsInside(nSrcX, nSrcY) || !xImageSrc.IsInside(nSrcX, nSrcY))
						{
							return;
						}

						TPixelSrc pixSrc;
						xImageSrc.ReadPixel2D(pixSrc, nSrcX, nSrcY);
						TPixelTrg pixTrg;

						float fLum = 0.2126f * float(Clu::MapPixelValue<TDataTrg>(pixSrc.r()))
							+ 0.7152f * float(Clu::MapPixelValue<TDataTrg>(pixSrc.g()))
							+ 0.0722f * float(Clu::MapPixelValue<TDataTrg>(pixSrc.b()));

						pixTrg.r() = Clu::CastFloatTo<TDataTrg>(fLum);

						if (TPixelTrg::AlphaCount > 0)
						{
							if (TPixelSrc::AlphaCount > 0)
							{
								pixTrg.a() = Clu::MapPixelValue<TDataTrg>(pixSrc.a());
							}
							else
							{
								pixTrg.a() = Clu::NumericLimits<typename TPixelTrg::TData>::Max();
							}
						}

						xImageTrg.WritePixel2D(pixTrg, nSrcX, nSrcY);
					}
				};

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>
				/// The conversion kernel entry function. This calls the corresponding function from the
				/// specialized structs.
				/// </summary>
				///
				/// <typeparam name="eConvType">		Type of the convert type. </typeparam>
				/// <typeparam name="TPixelTypeTrg">	Type of the pixel type trg. </typeparam>
				/// <typeparam name="TDataTypeTrg"> 	Type of the data type trg. </typeparam>
				/// <typeparam name="TPixelTypeSrc">	Type of the pixel type source. </typeparam>
				/// <typeparam name="TDataTypeSrc"> 	Type of the data type source. </typeparam>
				/// <typeparam name="TImageSrc">		Type of the image source. </typeparam>
				/// <param name="xImageTrg">	The image trg. </param>
				/// <param name="xImageSrc">	The image source. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<EConvertType eConvType
					, typename TPixelTypeTrg, typename TDataTypeTrg, typename TPixelTypeSrc, typename TDataTypeSrc
					, typename TImageTrg, typename TImageSrc>
				__global__ void Convert(TImageTrg xImageTrg, TImageSrc xImageSrc)
				{
					using TAlgo = typename SConvertAlgo<TPixelTypeTrg, TDataTypeTrg, TPixelTypeSrc, TDataTypeSrc, eConvType>;

					TAlgo::Convert(xImageTrg, xImageSrc);
				}




			} // namespace Kernel

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// The conversion struct that can be templated over the conversion algorithm type and the target
			/// and source pixel types.
			/// </summary>
			///
			/// <typeparam name="eConvType">	Type of the convert type. </typeparam>
			/// <typeparam name="TTrgPixel">	Type of the trg pixel. </typeparam>
			/// <typeparam name="TSrcPixel">	Type of the source pixel. </typeparam>
			/// <typeparam name="TImageSrc">	Type of the image source. </typeparam>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			template<EConvertType eConvType, typename TTrgPixel, typename TSrcPixel>
			struct SConvert
			{
				template<typename TImageTrg, typename TImageSrc>
				static bool Try_Select_TrgData(TImageTrg& xImageTrg, const TImageSrc& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (xImageTrg.Format().ePixelType != TTrgPixel::PixelTypeId
						|| xImageSrc.Format().ePixelType != TSrcPixel::PixelTypeId)
					{
						return false;
					}

					switch (xImageTrg.Format().eDataType)
					{
					case EDataType::UInt8:
						Select_SrcData<T_UInt8>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::UInt16:
						Select_SrcData<T_UInt16>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::UInt32:
						Select_SrcData<T_UInt32>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int8:
						Select_SrcData<T_Int8>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int16:
						Select_SrcData<T_Int16>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int32:
						Select_SrcData<T_Int32>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Single:
						Select_SrcData<T_Single>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Double:
						Select_SrcData<T_Double>(xImageTrg, xImageSrc, xDriverBase);
						break;

					default:
						throw CLU_EXCEPTION("Given target image data type unspported");
					}

					return true;
				}

				template<typename TImageSrc>
				static bool Try_Select_TrgData(Clu::Cuda::_CDeviceSurface& xImageTrg, const TImageSrc& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (xImageTrg.Format().ePixelType != TTrgPixel::PixelTypeId
						|| xImageSrc.Format().ePixelType != TSrcPixel::PixelTypeId)
					{
						return false;
					}

					switch (xImageTrg.Format().eDataType)
					{
					case EDataType::UInt8:
						Select_SrcData<T_UInt8>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::UInt16:
						Select_SrcData<T_UInt16>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::UInt32:
						Select_SrcData<T_UInt32>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int8:
						Select_SrcData<T_Int8>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int16:
						Select_SrcData<T_Int16>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Int32:
						Select_SrcData<T_Int32>(xImageTrg, xImageSrc, xDriverBase);
						break;

					case EDataType::Single:
						Select_SrcData<T_Single>(xImageTrg, xImageSrc, xDriverBase);
						break;

					default:
						throw CLU_EXCEPTION("Given target image data type unspported");
					}

					return true;
				}

				template<typename TTrgData, typename TImageTrg, typename TImageSrc>
				static void Select_SrcData(TImageTrg& xImageTrg, const TImageSrc& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					switch (xImageSrc.Format().eDataType)
					{
					case EDataType::UInt8:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt8>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::UInt16:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt16>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::UInt32:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt32>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int8:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int8>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int16:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int16>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int32:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int32>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Single:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Single>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Double:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Double>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					}
				}

				template<typename TTrgData, typename TImageTrg>
				static void Select_SrcData(TImageTrg& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					switch (xImageSrc.Format().eDataType)
					{
					case EDataType::UInt8:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt8>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::UInt16:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt16>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::UInt32:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_UInt32>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int8:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int8>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int16:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int16>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Int32:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Int32>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					case EDataType::Single:
						Kernel::Convert<eConvType, TTrgPixel, TTrgData, TSrcPixel, T_Single>
							<< <xDriverBase.BlocksInGrid(), xDriverBase.ThreadsPerBlock() >> >
							(xImageTrg, xImageSrc);
						break;

					}
				}

			};
			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>
			/// Process the Kernel. This function tests for pixel type combinations and executes the
			/// corresponding kernel that also transform the data type.
			/// </summary>
			///
			/// <typeparam name="TImageSrc">	Type of the image source. </typeparam>
			/// <param name="xImageTrg">	[in,out] The image trg. </param>
			/// <param name="xImageSrc">	The image source. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			
			template<typename TImageTrg, typename TImageSrc>
			struct SSelect
			{
				static void Process(TImageTrg& xImageTrg, const TImageSrc& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (SConvert<EConvertType::ColToCol, T_RGB, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToCol, T_RGBA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToLum, T_Lum, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					
					if (SConvert<EConvertType::LumToCol, T_RGB, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					
					if (SConvert<EConvertType::ColToLum, T_Lum, T_RGB>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					throw CLU_EXCEPTION("Given pixel type conversion not supported");
				}
			};

			template<typename TImageTrg>
			struct SSelect<TImageTrg, Clu::Cuda::_CDeviceSurface>
			{
				static void Process(TImageTrg& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (SConvert<EConvertType::ColToCol, T_RGB, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToCol, T_RGBA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToLum, T_Lum, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToLum, T_LumA, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToLum, T_LumA, T_LumA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToCol, T_RGB, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToCol, T_RGBA, T_LumA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::ColToLum, T_Lum, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToLum, T_LumA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					throw CLU_EXCEPTION("Given pixel type conversion not supported");
				}
			};

			template<typename TImageSrc>
			struct SSelect<Clu::Cuda::_CDeviceSurface, TImageSrc>
			{
				static void Process(Clu::Cuda::_CDeviceSurface& xImageTrg, const TImageSrc& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (SConvert<EConvertType::ColToCol, T_RGBA, T_RGB>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToCol, T_RGBA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToLum, T_Lum, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToLum, T_Lum, T_LumA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToCol, T_RGBA, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::ColToLum, T_Lum, T_RGB>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToLum, T_Lum, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToLum, T_LumA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					throw CLU_EXCEPTION("Given pixel type conversion not supported");
				}
			};

			template<>
			struct SSelect<Clu::Cuda::_CDeviceSurface, Clu::Cuda::_CDeviceSurface>
			{
				static void Process(Clu::Cuda::_CDeviceSurface& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc, CKernelDriverBase& xDriverBase)
				{
					if (SConvert<EConvertType::ColToCol, T_RGBA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToLum, T_Lum, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToLum, T_Lum, T_LumA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToLum, T_LumA, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::LumToCol, T_RGBA, T_Lum>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::LumToCol, T_RGBA, T_LumA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					if (SConvert<EConvertType::ColToLum, T_Lum, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;
					if (SConvert<EConvertType::ColToLum, T_LumA, T_RGBA>::Try_Select_TrgData(xImageTrg, xImageSrc, xDriverBase)) return;

					throw CLU_EXCEPTION("Given pixel type conversion not supported");
				}
			};

			template<typename TImageTrg, typename TImageSrc>
			void CDriver::Process(TImageTrg& xImageTrg, const TImageSrc& xImageSrc)
			{
				using _TImageTrg = typename TImageTrg::TDeviceObject;
				using _TImageSrc = typename TImageSrc::TDeviceObject;

				_TImageTrg xTrg = xImageTrg.AsDeviceObject();
				_TImageSrc xSrc = xImageSrc.AsDeviceObject();

				SSelect<_TImageTrg, _TImageSrc>::Process(xTrg, xSrc, *this);
			}

			template void CDriver::Process(Clu::Cuda::CDeviceSurface&, const Clu::Cuda::CDeviceLayerImage&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const Clu::Cuda::CDeviceLayerImage&);
			template void CDriver::Process(Clu::Cuda::CDeviceLayerImage&, const Clu::Cuda::CDeviceSurface&);
			template void CDriver::Process(Clu::Cuda::CDeviceLayerImage&, const Clu::Cuda::_CDeviceSurface&);

			template void CDriver::Process(Clu::Cuda::CDeviceSurface&, const Clu::Cuda::CDeviceImage&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const Clu::Cuda::CDeviceImage&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const Clu::Cuda::CDeviceSurface&);
			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const Clu::Cuda::_CDeviceSurface&);

			template void CDriver::Process(Clu::Cuda::CDeviceImage&, const Clu::Cuda::CDeviceImage&);
			template void CDriver::Process(Clu::Cuda::_CDeviceSurface&, const Clu::Cuda::_CDeviceSurface&);

			////////////////////////////////////////////////////////////////////////////////////////////////////
			/// <summary>	Configures. </summary>
			///
			/// <param name="xDevice">	The device. </param>
			/// <param name="xFormat">	Describes the format to use. </param>
			////////////////////////////////////////////////////////////////////////////////////////////////////

			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				EvalThreadConfig(xDevice, xFormat, 0, 0, NumberOfRegisters);
			}

		} // ImgProc
	} // Cuda
} // Clu


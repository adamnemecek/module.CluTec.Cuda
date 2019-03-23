////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      Homography.cu
//
// summary:   homography class
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

#include "Homography.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

#include "CluTec.Math/Static.Vector.h"

#include "CluTec.Cuda.Base/Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace Homography
		{
			namespace Kernel
			{
				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Homographies. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageOut">	The image out. </param>
				/// <param name="xImageIn"> 	The image in. </param>
				/// <param name="matHom">   	The matrix hom. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixel>
				__global__ void Homography(Clu::Cuda::_CDeviceImage xImageOut, Clu::Cuda::_CDeviceImage xImageIn,
					Clu::SMatrix<float, 3> matHom)
				{
					int nTrgX = int(blockIdx.x * blockDim.x + threadIdx.x);
					int nTrgY = int(blockIdx.y * blockDim.y + threadIdx.y);
					if (!xImageOut.IsInside(nTrgX, nTrgY))
					{
						return;
					}

					Clu::SVector3<float> vecPos(float(nTrgX), float(nTrgY), 1.0f);

					vecPos = matHom * vecPos;

					vecPos.x() /= vecPos.z();
					vecPos.y() /= vecPos.z();

					int nSrcX = int(floor(vecPos.x() + 0.5f));
					int nSrcY = int(floor(vecPos.y() + 0.5f));
					if (!xImageIn.IsInside(nSrcX, nSrcY))
					{
						return;
					}

					xImageOut.At<TPixel>(nTrgX, nTrgY) = xImageIn.At<TPixel>(nSrcX, nSrcY);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Homographies. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageOut">	The image out. </param>
				/// <param name="xImageIn"> 	The image in. </param>
				/// <param name="matHom">   	The matrix hom. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////


				template<typename TPixel>
				__global__ void Homography(Clu::Cuda::_CDeviceSurface xImageOut, Clu::Cuda::_CDeviceTexture xImageIn,
					Clu::SMatrix<float, 3> matHom);

				template<>
				__global__ void Homography<TPixel_RGBA_UInt8>(Clu::Cuda::_CDeviceSurface xImageOut, Clu::Cuda::_CDeviceTexture xImageIn,
					Clu::SMatrix<float, 3> matHom)
				{
					using TPixel = TPixel_RGBA_UInt8;
					using TNormalized = SPixelTypeInfo<TPixel>::TNormalized;
					using TElement = SPixelTypeInfo<TPixel>::TElement;
					using TComponent = TPixel::TData;

					int nTrgX = int(blockIdx.x * blockDim.x + threadIdx.x);
					int nTrgY = int(blockIdx.y * blockDim.y + threadIdx.y);
					if (!xImageOut.IsInside(nTrgX, nTrgY))
					{
						return;
					}

					Clu::SVector3<float> vecPos(float(nTrgX), float(nTrgY), 1.0f);

					vecPos = matHom * vecPos;

					vecPos.x() /= vecPos.z();
					vecPos.y() /= vecPos.z();

					int nSrcX = int(floor(vecPos.x() + 0.5f));
					int nSrcY = int(floor(vecPos.y() + 0.5f));
					if (!xImageIn.IsInside(nSrcX, nSrcY))
					{
						return;
					}

					TNormalized pixSrc = xImageIn.TexNorm2D<TPixel>(vecPos.x(), vecPos.y());
					TElement pixTrg = NormToRawPix<TPixel>(pixSrc);

					xImageOut.Write2D(pixTrg, nTrgX, nTrgY);
				}

			}



			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat)
			{
				EvalThreadConfig(xDevice, xFormat, 0, 0, NumberOfRegisters);
			}


			void CDriver::Process(Clu::Cuda::CDeviceImage& xImageOut, const Clu::Cuda::CDeviceImage& xImageIn,
				const Clu::SMatrix<float, 3>& matHom)
			{
				if (!xImageOut.IsEqualFormat(xImageIn.Format()))
				{
					throw CLU_EXCEPTION("Input and output images are not of the same format");
				}

				if (xImageIn.IsOfType<Clu::TPixel_RGB_UInt8>())
				{
					Kernel::Homography<Clu::TPixel_RGB_UInt8> 
						CLU_KERNEL_CONFIG()
						(xImageOut, xImageIn, matHom);
				}
				else
				{
					throw CLU_EXCEPTION("Given color pixel type not supported");
				}
			}


			void CDriver::Process(Clu::Cuda::CDeviceSurface& xImageOut, const Clu::Cuda::CDeviceTexture& xImageIn,
				const Clu::SMatrix<float, 3>& matHom)
			{
				if (!xImageOut.IsEqualFormat(xImageIn.Format()))
				{
					throw CLU_EXCEPTION("Input and output images are not of the same format");
				}

				if (xImageIn.IsOfType<Clu::TPixel_RGBA_UInt8>())
				{
					Kernel::Homography<Clu::TPixel_RGBA_UInt8> 
						CLU_KERNEL_CONFIG()
						(xImageOut, xImageIn, matHom);
				}
				else
				{
					throw CLU_EXCEPTION("Given color pixel type not supported");
				}
			}


		} // ImgProc
	} // Cuda
} // Clu


////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      RectifyImage.cu
//
// summary:   rectify image class
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

#include "RectifyImage.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"

#include "CluTec.Math/Static.Vector.h"
#include "CluTec.Math/Static.Matrix.IO.h"


//#define CLU_DEBUG_KERNEL
#include "CluTec.Cuda.Base/Kernel.Debug.h"

namespace Clu
{
	namespace Cuda
	{
		namespace RectifyImage
		{
			namespace Kernel
			{
				using namespace Clu::Cuda::Kernel;

				struct Const
				{
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
				};

				__constant__ _SParameter c_xPars;

				template<typename TPixel>
				__device__ void CopyPixel(int iTrgX, int iTrgY, const TVec2& vSrcPos, Clu::Cuda::_CDeviceSurface& xImageTrg, Clu::Cuda::_CDeviceSurface& xImageSrc)
				{
					using TData = typename TPixel::TData;

					TVec2 vSrcPosTL;
					vSrcPosTL = Floor(vSrcPos);

					_SVector<int, 2> vPixNearest;
					vPixNearest.SetElements((int)floor(vSrcPos[0] + 0.5f), (int)floor(vSrcPos[1] + 0.5f));

					TPixel pixValue;

					// Test whether 2x2 block of pixels vSrcPos lies inside of is completely inside the source image.
					if (xImageSrc.IsRectInside((int)vSrcPosTL[0], (int)vSrcPosTL[1], 2, 2))
					{
						TVec2 vPixFrac;
						int iX, iY;
						iX = (int)vSrcPosTL[0];
						iY = (int)vSrcPosTL[1];

						vPixFrac = vSrcPos - vSrcPosTL;
						float fValue = (1.0f - vPixFrac[0]) * (1.0f - vPixFrac[1]) * Clu::ToNormFloat<float>(xImageSrc.ReadPixel2D<TPixel>(iX, iY).r());
						fValue += vPixFrac[0] * (1.0f - vPixFrac[1]) * Clu::ToNormFloat<float>(xImageSrc.ReadPixel2D<TPixel>(iX + 1, iY).r());
						fValue += (1.0f - vPixFrac[0]) * vPixFrac[1] * Clu::ToNormFloat<float>(xImageSrc.ReadPixel2D<TPixel>(iX, iY + 1).r());
						fValue += vPixFrac[0] * vPixFrac[1] * Clu::ToNormFloat<float>(xImageSrc.ReadPixel2D<TPixel>(iX + 1, iY + 1).r());

						pixValue.r() = Clu::NormFloatTo<TData>(fValue);
					}
					// Otherwise test whether at least the nearest pixel is inside
					else if (xImageSrc.IsInside(vPixNearest[0], vPixNearest[1]))
					{
						pixValue = xImageSrc.ReadPixel2D<TPixel>(vPixNearest[0], vPixNearest[1]);
					}
					else
					{
						// Otherwise the source image has no corresponding pixel
						pixValue.r() = TData(0);
					}

					xImageTrg.WritePixel2D<TPixel>(pixValue, iTrgX, iTrgY);
				}

				__device__ __forceinline__ void MapPixelTrgToSrc(TVec2& vSrcPos, int iTrgX, int iTrgY)
				{
					TVec3 vPos;
					// Get metric coordinates of target pixel in rectified sensor frame
					c_xPars.xCameraRectified.Sensor().Map_PixelIX_to_SensorM(vPos, iTrgX, iTrgY);

					// Map those coordinates into the original camera frame
					vPos = c_xPars._xFrame_r_s.MapOutOfFrame(vPos);
					//vPos = c_xPars.xCameraRectified.Sensor().FrameM_s_w().MapOutOfFrame(vPos);
					//vPos = c_xPars.xCamera.Sensor().FrameM_s_w().MapIntoFrame(vPos);

					// Project point onto image plane and get sub-pixel position
					TValue dDepth_s;
					c_xPars.xCamera.Project_SensorM_to_PixelF(vSrcPos[0], vSrcPos[1], dDepth_s, vPos);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Rectify image. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageTrg">	The image trg. </param>
				/// <param name="xImageSrc">	The image source. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixel>
				__global__ void Rectify(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					int iTrgX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int iTrgY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageTrg.IsInside(iTrgX, iTrgY))
					{
						return;
					}

					TVec2 vSrcPos;
					MapPixelTrgToSrc(vSrcPos, iTrgX, iTrgY);
					CopyPixel<TPixel>(iTrgX, iTrgY, vSrcPos, xImageTrg, xImageSrc);
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Rectify and undistort image. </summary>
				///
				/// <typeparam name="TPixel">	Type of the pixel. </typeparam>
				/// <param name="xImageTrg">	The image trg. </param>
				/// <param name="xImageSrc">	The image source. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixel>
				__global__ void RectifyUndistort(Clu::Cuda::_CDeviceSurface xImageTrg, Clu::Cuda::_CDeviceSurface xImageSrc)
				{
					int iTrgX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int iTrgY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);
					if (!xImageTrg.IsInside(iTrgX, iTrgY))
					{
						return;
					}

					TVec2 vSrcPos;
					MapPixelTrgToSrc(vSrcPos, iTrgX, iTrgY);

					TVec2 vDistPos;
					c_xPars.xCamera.Distortion().DistortPX(vDistPos, vSrcPos, c_xPars.xCamera);

					//Debug::Run([&]()
					//{
					//	if (Debug::IsBlock(10, 10))
					//	{
					//		printf("(%g, %g) -> (%g, %g)\n", vSrcPos[0], vSrcPos[1], vDistPos[0], vDistPos[1]);
					//	}
					//});

					CopyPixel<TPixel>(iTrgX, iTrgY, vDistPos, xImageTrg, xImageSrc);
				}

			}



			void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
				, const _SParameter& xPars)
			{
				EvalThreadConfigBlockSize(xDevice, xFormat
					, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
					, 0, 0, 0, 0 // Offsets
					, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
					, NumberOfRegisters
					, false // Use also partial blocks
					);

				m_xPars = xPars;

				// Calculate the combined frame of first mapping out of the rectified camera frame
				// and then into the original camera frame.
				m_xPars.xCameraRectified.Sensor().FrameM_s_w().ConcatFrames_Out_In(m_xPars._xFrame_r_s, m_xPars.xCamera.Sensor().FrameM_s_w());

				
				//CLU_LOG(CLU_S "Camera:\n" << ToString(m_xPars.xCamera.Sensor().FrameM_s_w().Basis_l_r(), "%6.4f").c_str());
				//CLU_LOG(CLU_S "Rectified Camera:\n" << ToString(m_xPars.xCameraRectified.Sensor().FrameM_s_w().Basis_l_r(), "%6.4f").c_str());
				//CLU_LOG(CLU_S "Out-In:\n" << ToString(m_xPars._xFrame_r_s.Basis_l_r(), "%6.4f").c_str());
			}


			void CDriver::Process(Clu::Cuda::_CDeviceSurface& xImageTrg, const Clu::Cuda::_CDeviceSurface& xImageSrc)
			{
				if (!xImageTrg.IsEqualType(xImageSrc.Format()))
				{
					throw CLU_EXCEPTION("Input and output images do not have the same type");
				}

				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);
				
				if (xImageSrc.IsOfType<Clu::TPixel_Lum_UInt8>()
					&& xImageTrg.IsOfType<Clu::TPixel_Lum_UInt8>())
				{
					//Kernel::Rectify<Clu::TPixel_Lum_UInt8>
					Kernel::RectifyUndistort<Clu::TPixel_Lum_UInt8>
						CLU_KERNEL_CONFIG()
						(xImageTrg, xImageSrc);
				}
				else
				{
					throw CLU_EXCEPTION("Pixel types of given images not supported");
				}

			}





		} // ImgProc
	} // Cuda
} // Clu


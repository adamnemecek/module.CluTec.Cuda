////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgProc
// file:      RectifyImage.h
//
// summary:   Declares the rectify image class
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
#include "CluTec.Math/Static.Matrix.h"
#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/DeviceTexture.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.ImgProc/Camera.Pinhole.h"
#include "CluTec.ImgProc/Camera.Distortion.h"

namespace Clu
{
	namespace Cuda
	{
		namespace RectifyImage
		{
			using TValue = float;
			using TPinhole = Clu::Camera::_CPinhole<TValue>;
			using TDistort = Clu::Camera::_CDistortion<TValue, Clu::Camera::EDistStyle::OpenCV>;

			using TVec3 = Clu::_SVector<TValue, 3>;
			using TVec2 = Clu::_SVector<TValue, 2>;
			using TFrame3D = Clu::_CFrame3D<TValue>;

			struct _SParameter
			{
				TPinhole xCamera;
				TPinhole xCameraRectified;

				//Calculated values
				// 
				/// <summary> The combines frame that maps a point from the rectified frame
				/// 		  to the camera sensor frame. This avoids mapping first to world
				/// 		  coordinates and then back.</summary>
				TFrame3D _xFrame_r_s;
			};

			class CDriver : public CKernelDriverBase
			{
			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 22;
#else
				static const size_t NumberOfRegisters = 11;
#endif

			protected:
				_SParameter m_xPars;

			public:
				CDriver()
					: CKernelDriverBase("Rectify Image")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
					, const _SParameter& xPars);

				void Process(Clu::Cuda::_CDeviceSurface& xImageTrg
					, const Clu::Cuda::_CDeviceSurface& xImageSrc);



			};

		} // ImgProc
	} // Cuda
} // Clu


////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      CreateDispMipMaps.h
//
// summary:   Declares the create disp mip maps class
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
#include "CluTec.Cuda.Base/DeviceSurfMipMap.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"

namespace Clu
{
	namespace Cuda
	{
		namespace CreateDispMipMaps
		{
			using TPixelDisp = _SDisparityConfig::TPixelEx;
			using TPixelDispF = _SDisparityConfig::TPixelF;
			using TDisp = _SDisparityConfig::TDisp;

			class CDriver : public Clu::Cuda::CKernelDriverBase
			{
			public:
				using TId = Clu::Cuda::CKernelDriverBase::TId;

			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 50;
#else
				static const size_t NumberOfRegisters = 11;
#endif
				static const TId KernelSmoothHorizontal = 0;
				static const TId KernelSmoothVertical = 1;
				static const TId KernelCopyDispChannel = 2;

			protected:

		public:
				CDriver()
					: CKernelDriverBase("Create Disparity Mip Maps")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

				void Run(Clu::Cuda::CDeviceSurfMipMap& xImage, Clu::Cuda::CDeviceSurface& xTemp, const Clu::Cuda::_CDeviceSurface& surfDisp, int iLevelCount = 0);


			};

		} // ImgProc
	} // Cuda
} // Clu


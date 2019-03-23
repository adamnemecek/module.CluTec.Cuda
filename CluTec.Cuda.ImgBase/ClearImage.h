////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      ClearImage.h
//
// summary:   Declares the clear image class
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
#include "CluTec.Cuda.Base/DeviceImage.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"

namespace Clu
{
	namespace Cuda
	{
		namespace ClearImage
		{

			class CDriver : public CKernelDriverBase
			{
			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 12;
#else
				static const size_t NumberOfRegisters = 11;
#endif


			public:
				CDriver()
					: CKernelDriverBase("Clear Image")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

				template<typename TPixel>
				void Process(Clu::Cuda::CDeviceImage& xImage, const TPixel& pixClear);

				template<typename TPixel>
				void Process(Clu::Cuda::_CDeviceSurface& xImage, const TPixel& pixClear);


			};

		} // ImgProc
	} // Cuda
} // Clu


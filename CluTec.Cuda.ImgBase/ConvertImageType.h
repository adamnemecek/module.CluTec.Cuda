////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.ImgBase
// file:      ConvertImageType.h
//
// summary:   Declares the convert image type class
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
#include "CluTec.Cuda.Base/KernelDriverBase.h"

namespace Clu
{
	namespace Cuda
	{
		namespace ConvertImageType
		{

			class CDriver : public CKernelDriverBase
			{
			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 20;
#else
				static const size_t NumberOfRegisters = 11;
#endif

			protected:

			public:
				CDriver()
					: CKernelDriverBase("Convert Image Type")
				{}

				~CDriver()
				{}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat);

				template<typename TImageTrg, typename TImageSrc>
				void Process(TImageTrg& xImageTrg, const TImageSrc& xImageSrc);

			};


		} // ImgProc
	} // Cuda
} // Clu


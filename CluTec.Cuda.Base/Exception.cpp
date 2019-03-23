////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Exception.cpp
//
// summary:   Implements the exception class
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

#include "Exception.h"

namespace Clu
{
	namespace ExceptionType
	{

		// {FFF9A736-C596-4CD5-B3DB-FD306CFB3FF4}
		const CGuid Cuda::Guid = CGuid(0xfff9a736, 0xc596, 0x4cd5, 0xb3, 0xdb, 0xfd, 0x30, 0x6c, 0xfb, 0x3f, 0xf4);
		const char* Cuda::TypeName = "Cuda Runtime Error";

	}
}

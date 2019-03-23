////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      DisparityId.h
//
// summary:   Declares the disparity identifier class
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

namespace Clu
{
	namespace Cuda
	{
		enum class EDisparityId : unsigned short
		{
			First = 1,
			Last = 0xFFF0 - 1,

			Unknown = 0,
			CannotEvaluate = 0xFFFF,
			Saturated = 0xFFFF - 1,
			NotFound = 0xFFFF - 2,
			NotSpecific = 0xFFFF - 3,
			NotUnique = 0xFFFF - 4,
			Inconsistent = 0xFFFF - 5,
		};

	}
}
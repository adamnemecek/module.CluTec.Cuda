////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      TextureType.h
//
// summary:   Declares the texture type class
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
#include "Api.h"


namespace Clu
{
	namespace Cuda
	{
		struct STextureType
		{
			ETextureAddressMode eAddressMode;
			ETextureFilterMode eFilterMode;
			ETextureReadMode eReadMode;
			bool bNormalizedCoords;

			STextureType()
			{
				Reset();
			}

			STextureType(ETextureAddressMode _eAddressMode
				, ETextureFilterMode _eFilterMode
				, ETextureReadMode _eReadMode
				, bool _bNormalizedCoords)
			{
				Set(_eAddressMode, _eFilterMode, _eReadMode, _bNormalizedCoords);
			}

			bool operator==(const STextureType& xType)
			{
				return (
					eAddressMode == xType.eAddressMode
					&& eFilterMode == xType.eFilterMode
					&& eReadMode == xType.eReadMode
					&& bNormalizedCoords == xType.bNormalizedCoords);
			}

			bool operator!=(const STextureType& xType)
			{
				return !(*this == xType);
			}

			void Set(ETextureAddressMode _eAddressMode
					, ETextureFilterMode _eFilterMode
					, ETextureReadMode _eReadMode
					, bool _bNormalizedCoords)
			{
				eAddressMode = _eAddressMode;
				eFilterMode = _eFilterMode;
				eReadMode = _eReadMode;
				bNormalizedCoords = _bNormalizedCoords;
			}

			void Reset()
			{
				eAddressMode = ETextureAddressMode::Clamp;
				eFilterMode = ETextureFilterMode::Linear;
				eReadMode = ETextureReadMode::NormalizedFloat;
				bNormalizedCoords = false;
			}

			operator cudaTextureDesc() const;
		};
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      TextureType.cpp
//
// summary:   Implements the texture type class
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

#include "TextureType.h"

namespace Clu
{
	namespace Cuda
	{

		STextureType::operator cudaTextureDesc() const
		{
			cudaTextureDesc xT;
			
			memset(&xT, 0, sizeof(cudaTextureDesc));
			xT.addressMode[0] = cudaTextureAddressMode(eAddressMode);
			xT.addressMode[1] = cudaTextureAddressMode(eAddressMode);
			xT.filterMode = cudaTextureFilterMode(eFilterMode);
			xT.readMode = cudaTextureReadMode(eReadMode);
			xT.normalizedCoords = (bNormalizedCoords ? 1 : 0);

			return xT;
		}
	}
}
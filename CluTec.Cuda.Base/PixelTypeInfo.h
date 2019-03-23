////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      PixelTypeInfo.h
//
// summary:   Declares the pixel type information class
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
#include "CluTec.Types1/Pixel.h"

namespace Clu
{
	namespace Cuda
	{
		template<typename TPixel>
		struct SPixelTypeInfo
		{
		};

#define DECL_PTI(theType, theElType, theNormType) \
		template<> \
		struct SPixelTypeInfo<theType> \
				{ \
			using TComponent = theType::TData; \
			using TElement = theElType; \
			using TNormalized = theNormType; \
		}


		DECL_PTI(TPixel_Lum_Int8, char1, float1);
		DECL_PTI(TPixel_Lum_Int16, short1, float1);
		DECL_PTI(TPixel_Lum_Int32, int1, float1);
		DECL_PTI(TPixel_Lum_UInt8, uchar1, float1);
		DECL_PTI(TPixel_Lum_UInt16, ushort1, float1);
		DECL_PTI(TPixel_Lum_UInt32, uint1, float1);
		DECL_PTI(TPixel_Lum_Single, float1, float1);
		DECL_PTI(TPixel_Lum_Double, double1, double1);

		DECL_PTI(TPixel_LumA_Int8, char2, float2);
		DECL_PTI(TPixel_LumA_Int16, short2, float2);
		DECL_PTI(TPixel_LumA_Int32, int2, float2);
		DECL_PTI(TPixel_LumA_UInt8, uchar2, float2);
		DECL_PTI(TPixel_LumA_UInt16, ushort2, float2);
		DECL_PTI(TPixel_LumA_UInt32, uint2, float2);
		DECL_PTI(TPixel_LumA_Single, float2, float2);
		DECL_PTI(TPixel_LumA_Double, double2, double2);

		DECL_PTI(TPixel_RGB_Int8, char3, float3);
		DECL_PTI(TPixel_RGB_Int16, short3, float3);
		DECL_PTI(TPixel_RGB_Int32, int3, float3);
		DECL_PTI(TPixel_RGB_UInt8, uchar3, float3);
		DECL_PTI(TPixel_RGB_UInt16, ushort3, float3);
		DECL_PTI(TPixel_RGB_UInt32, uint3, float3);
		DECL_PTI(TPixel_RGB_Single, float3, float3);
		DECL_PTI(TPixel_RGB_Double, double3, double3);

		DECL_PTI(TPixel_BGR_Int8, char3, float3);
		DECL_PTI(TPixel_BGR_Int16, short3, float3);
		DECL_PTI(TPixel_BGR_Int32, int3, float3);
		DECL_PTI(TPixel_BGR_UInt8, uchar3, float3);
		DECL_PTI(TPixel_BGR_UInt16, ushort3, float3);
		DECL_PTI(TPixel_BGR_UInt32, uint3, float3);
		DECL_PTI(TPixel_BGR_Single, float3, float3);
		DECL_PTI(TPixel_BGR_Double, double3, double3);

		DECL_PTI(TPixel_RGBA_Int8, char4, float4);
		DECL_PTI(TPixel_RGBA_Int16, short4, float4);
		DECL_PTI(TPixel_RGBA_Int32, int4, float4);
		DECL_PTI(TPixel_RGBA_UInt8, uchar4, float4);
		DECL_PTI(TPixel_RGBA_UInt16, ushort4, float4);
		DECL_PTI(TPixel_RGBA_UInt32, uint4, float4);
		DECL_PTI(TPixel_RGBA_Single, float4, float4);
		DECL_PTI(TPixel_RGBA_Double, double4, double4);

		DECL_PTI(TPixel_BGRA_Int8, char4, float4);
		DECL_PTI(TPixel_BGRA_Int16, short4, float4);
		DECL_PTI(TPixel_BGRA_Int32, int4, float4);
		DECL_PTI(TPixel_BGRA_UInt8, uchar4, float4);
		DECL_PTI(TPixel_BGRA_UInt16, ushort4, float4);
		DECL_PTI(TPixel_BGRA_UInt32, uint4, float4);
		DECL_PTI(TPixel_BGRA_Single, float4, float4);
		DECL_PTI(TPixel_BGRA_Double, double4, double4);


#undef DECL_PTI

	}
}
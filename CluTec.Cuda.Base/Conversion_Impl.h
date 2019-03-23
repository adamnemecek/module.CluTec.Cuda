////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Conversion_Impl.h
//
// summary:   Declares the conversion implementation class
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
#include "Conversion.h"

namespace Clu
{
	namespace Cuda
	{
		template<>
		__CUDA_HDI__ uchar4 ConvertTo<uchar4>(const uchar1& xValue)
		{
			return ::make_uchar4(xValue.x, xValue.x, xValue.x, 255);
		}


		template<>
		__CUDA_HDI__ uchar1 Make<uchar1>(unsigned char ucVal1)
		{
			return ::make_uchar1(ucVal1);
		}

		template<>
		__CUDA_HDI__ uchar1 Make<uchar1>(int iVal1)
		{
			return ::make_uchar1((unsigned char)iVal1);
		}

		template<>
		__CUDA_HDI__ uchar1 Make<uchar1>(unsigned int iVal1)
		{
			return ::make_uchar1((unsigned char)iVal1);
		}

		template<>
		__CUDA_HDI__ uchar4 Make<uchar4>(unsigned char ucVal1, unsigned char ucVal2, unsigned char ucVal3, unsigned char ucVal4)
		{
			return ::make_uchar4(ucVal1, ucVal2, ucVal3, ucVal4);
		}


		template<>
		__CUDA_HDI__ uchar4 Make<uchar4>(int iVal1, int iVal2, int iVal3, int iVal4)
		{
			using T = unsigned char;

			return ::make_uchar4(T(iVal1), T(iVal2), T(iVal3), T(iVal4));
		}

		template<>
		__CUDA_HDI__ ushort4 Make<ushort4>(unsigned short ucVal1, unsigned short ucVal2, unsigned short ucVal3, unsigned short ucVal4)
		{
			return ::make_ushort4(ucVal1, ucVal2, ucVal3, ucVal4);
		}


		template<>
		__CUDA_HDI__ Clu::TPixel_RGBA_UInt8 Make<Clu::TPixel_RGBA_UInt8>(unsigned char ucVal1, unsigned char ucVal2, unsigned char ucVal3, unsigned char ucVal4)
		{
			Clu::TPixel_RGBA_UInt8 pixValue;

			pixValue.r() = ucVal1;
			pixValue.g() = ucVal2;
			pixValue.b() = ucVal3;
			pixValue.a() = ucVal4;

			return pixValue;
		}

		template<>
		__CUDA_HDI__ Clu::TPixel_RGBA_UInt8 Make<Clu::TPixel_RGBA_UInt8>(int ucVal1, int ucVal2, int ucVal3, int ucVal4)
		{
			Clu::TPixel_RGBA_UInt8 pixValue;
			using T = unsigned char;

			pixValue.r() = T(ucVal1);
			pixValue.g() = T(ucVal2);
			pixValue.b() = T(ucVal3);
			pixValue.a() = T(ucVal4);

			return pixValue;
		}


#define CLU_MAKE_ZERO_1(theType) \
		template<> \
		__CUDA_HDI__ theType MakeZero<theType>() \
		{ \
			return ::make_##theType(0); \
		}

		CLU_MAKE_ZERO_1(char1);
		CLU_MAKE_ZERO_1(short1);
		CLU_MAKE_ZERO_1(int1);

		CLU_MAKE_ZERO_1(uchar1);
		CLU_MAKE_ZERO_1(ushort1);
		CLU_MAKE_ZERO_1(uint1);

		CLU_MAKE_ZERO_1(float1);
		CLU_MAKE_ZERO_1(double1);

#undef CLU_MAKE_ZERO_1

#define CLU_MAKE_ZERO_2(theType) \
		template<> \
		__CUDA_HDI__ theType MakeZero<theType>() \
		{ \
			return ::make_##theType(0, 0); \
		}

		CLU_MAKE_ZERO_2(char2);
		CLU_MAKE_ZERO_2(short2);
		CLU_MAKE_ZERO_2(int2);

		CLU_MAKE_ZERO_2(uchar2);
		CLU_MAKE_ZERO_2(ushort2);
		CLU_MAKE_ZERO_2(uint2);

		CLU_MAKE_ZERO_2(float2);
		CLU_MAKE_ZERO_2(double2);

#undef CLU_MAKE_ZERO_2

#define CLU_MAKE_ZERO_3(theType) \
		template<> \
		__CUDA_HDI__ theType MakeZero<theType>() \
		{ \
			return ::make_##theType(0, 0, 0); \
		}

		CLU_MAKE_ZERO_3(char3);
		CLU_MAKE_ZERO_3(short3);
		CLU_MAKE_ZERO_3(int3);

		CLU_MAKE_ZERO_3(uchar3);
		CLU_MAKE_ZERO_3(ushort3);
		CLU_MAKE_ZERO_3(uint3);

		CLU_MAKE_ZERO_3(float3);
		CLU_MAKE_ZERO_3(double3);

#undef CLU_MAKE_ZERO_3

#define CLU_MAKE_ZERO_4(theType) \
		template<> \
		__CUDA_HDI__ theType MakeZero<theType>() \
		{ \
			return ::make_##theType(0, 0, 0, 0); \
		}

		CLU_MAKE_ZERO_4(char4);
		CLU_MAKE_ZERO_4(short4);
		CLU_MAKE_ZERO_4(int4);

		CLU_MAKE_ZERO_4(uchar4);
		CLU_MAKE_ZERO_4(ushort4);
		CLU_MAKE_ZERO_4(uint4);

		CLU_MAKE_ZERO_4(float4);
		CLU_MAKE_ZERO_4(double4);

#undef CLU_MAKE_ZERO_4


#define CLU_ASSIGN1(theType) \
		template<> \
		__CUDA_HDI__ void Assign(volatile theType& xTrg, const theType& xSrc) \
		{ \
			xTrg.x = xSrc.x; \
		} \
		template<> \
			__CUDA_HDI__ void Assign(theType& xTrg, const theType& xSrc) \
		{ \
		xTrg.x = xSrc.x; \
		}


		CLU_ASSIGN1(char1);
		CLU_ASSIGN1(short1);
		CLU_ASSIGN1(int1);

		CLU_ASSIGN1(uchar1);
		CLU_ASSIGN1(ushort1);
		CLU_ASSIGN1(uint1);

		CLU_ASSIGN1(float1);
		CLU_ASSIGN1(double1);

#undef CLU_ASSIGN1


#define CLU_ASSIGN2(theType) \
		template<> \
		__CUDA_HDI__ void Assign(volatile theType& xTrg, const theType& xSrc) \
		{ \
			xTrg.x = xSrc.x; \
			xTrg.y = xSrc.y; \
		} \
		template<> \
			__CUDA_HDI__ void Assign(theType& xTrg, const theType& xSrc) \
		{ \
		xTrg.x = xSrc.x; \
		xTrg.y = xSrc.y; \
		}

		CLU_ASSIGN2(char2);
		CLU_ASSIGN2(short2);
		CLU_ASSIGN2(int2);

		CLU_ASSIGN2(uchar2);
		CLU_ASSIGN2(ushort2);
		CLU_ASSIGN2(uint2);

		CLU_ASSIGN2(float2);
		CLU_ASSIGN2(double2);


#undef CLU_ASSIGN2

#define CLU_ASSIGN3(theType) \
		template<> \
		__CUDA_HDI__ void Assign(volatile theType& xTrg, const theType& xSrc) \
		{ \
			xTrg.x = xSrc.x; \
			xTrg.y = xSrc.y; \
			xTrg.z = xSrc.z; \
		} \
		template<> \
			__CUDA_HDI__ void Assign(theType& xTrg, const theType& xSrc) \
		{ \
		xTrg.x = xSrc.x; \
		xTrg.y = xSrc.y; \
		xTrg.z = xSrc.z; \
		}

		CLU_ASSIGN3(char3);
		CLU_ASSIGN3(short3);
		CLU_ASSIGN3(int3);

		CLU_ASSIGN3(uchar3);
		CLU_ASSIGN3(ushort3);
		CLU_ASSIGN3(uint3);

		CLU_ASSIGN3(float3);
		CLU_ASSIGN3(double3);


#undef CLU_ASSIGN3

#define CLU_ASSIGN4(theType) \
		template<> \
		__CUDA_HDI__ void Assign(volatile theType& xTrg, const theType& xSrc) \
						{ \
			xTrg.x = xSrc.x; \
			xTrg.y = xSrc.y; \
			xTrg.z = xSrc.z; \
			xTrg.w = xSrc.w; \
						} \
		template<> \
		__CUDA_HDI__ void Assign(theType& xTrg, const theType& xSrc) \
		{ \
		xTrg.x = xSrc.x; \
		xTrg.y = xSrc.y; \
		xTrg.z = xSrc.z; \
		xTrg.w = xSrc.w; \
		}

		CLU_ASSIGN4(char4);
		CLU_ASSIGN4(short4);
		CLU_ASSIGN4(int4);

		CLU_ASSIGN4(uchar4);
		CLU_ASSIGN4(ushort4);
		CLU_ASSIGN4(uint4);

		CLU_ASSIGN4(float4);
		CLU_ASSIGN4(double4);


#undef CLU_ASSIGN4


	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      ChannelFormat.h
//
// summary:   Declares the channel format class
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
#include <utility>

#include "cuda_runtime.h"
#include "CluTec.Types1/ImageType.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Base/Exception.h"

namespace Clu
{
	namespace Cuda
	{

		template<Clu::EDataType t_eDataType>
		struct SChannelKind
		{
			using TDataTypeInfo = SDataTypeInfo<t_eDataType>;
			using TData = typename TDataTypeInfo::TData;

			static cudaChannelFormatKind Kind_Impl(std::true_type, std::true_type)
			{
				return cudaChannelFormatKindSigned;
			}

			static cudaChannelFormatKind Kind_Impl(std::true_type, std::false_type)
			{
				return cudaChannelFormatKindUnsigned;
			}

			static cudaChannelFormatKind Kind_Impl(std::false_type, std::true_type)
			{
				return cudaChannelFormatKindFloat;
			}

			static cudaChannelFormatKind Kind()
			{
				return Kind_Impl(std::is_integral<TData>(), std::is_signed<TData>());
			}
		};


		class CChannelFormat
		{
		private:
			cudaChannelFormatDesc m_xCfd;

		public:
			CChannelFormat()
			{
				Clear();
			}

			CChannelFormat(const CChannelFormat& xFormat)
			{
				m_xCfd = xFormat.m_xCfd;
			}

			~CChannelFormat()
			{}

			operator cudaChannelFormatDesc() const
			{
				return m_xCfd;
			}

			operator const cudaChannelFormatDesc*() const
			{
				return &m_xCfd;
			}

			void Clear()
			{
				m_xCfd.f = cudaChannelFormatKindNone;
				m_xCfd.x = 0;
				m_xCfd.y = 0;
				m_xCfd.z = 0;
				m_xCfd.w = 0;
			}

		protected:
			template<typename TPixelType>
			void _SetChannelBits(int iBitsPerChannel)
			{
				m_xCfd.x = (TPixelType::ChannelCount > 0 ? iBitsPerChannel : 0);
				m_xCfd.y = (TPixelType::ChannelCount > 1 ? iBitsPerChannel : 0);
				m_xCfd.z = (TPixelType::ChannelCount > 2 ? iBitsPerChannel : 0);
				m_xCfd.w = (TPixelType::ChannelCount > 3 ? iBitsPerChannel : 0);
			}

			template<typename TDataType>
			void _SetChannelFormat(Clu::EPixelType ePixelType)
			{
				int iBits = TDataType::Bits;

				m_xCfd.f = SChannelKind<TDataType::DataTypeId>::Kind();

				switch (ePixelType)
				{
				case Clu::EPixelType::RGB:
					_SetChannelBits<T_RGB>(iBits);
					break;
				case Clu::EPixelType::RGBA:
					_SetChannelBits<T_RGBA>(iBits);
					break;
				case Clu::EPixelType::BGR:
					_SetChannelBits<T_BGR>(iBits);
					break;
				case Clu::EPixelType::BGRA:
					_SetChannelBits<T_BGRA>(iBits);
					break;
				case Clu::EPixelType::Lum:
					_SetChannelBits<T_Lum>(iBits);
					break;
				case Clu::EPixelType::LumA:
					_SetChannelBits<T_LumA>(iBits);
					break;
				default:
					throw CLU_EXCEPTION("Pixel type not supported");
				}
			}

		public:
			void Set(const Clu::SImageType &xType)
			{
				switch (xType.eDataType)
				{
				case Clu::EDataType::Int8:
					_SetChannelFormat<T_Int8>(xType.ePixelType);
					break;
				case Clu::EDataType::UInt8:
					_SetChannelFormat<T_UInt8>(xType.ePixelType);
					break;
				case Clu::EDataType::Int16:
					_SetChannelFormat<T_Int16>(xType.ePixelType);
					break;
				case Clu::EDataType::UInt16:
					_SetChannelFormat<T_UInt16>(xType.ePixelType);
					break;
				case Clu::EDataType::Int32:
					_SetChannelFormat<T_Int32>(xType.ePixelType);
					break;
				case Clu::EDataType::UInt32:
					_SetChannelFormat<T_UInt32>(xType.ePixelType);
					break;
				case Clu::EDataType::Single:
					_SetChannelFormat<T_Single>(xType.ePixelType);
					break;
				case Clu::EDataType::Double:
					_SetChannelFormat<T_Double>(xType.ePixelType);
					break;
				default:
					throw CLU_EXCEPTION("Data type not supported");
				}

			}
		};
	}
}

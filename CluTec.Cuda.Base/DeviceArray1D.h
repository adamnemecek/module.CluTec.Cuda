////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceArray1D.h
//
// summary:   Declares the device array 1 d class
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
#include "CluTec.Base/Exception.h"

namespace Clu
{
	namespace Cuda
	{

		template<typename _TValue>
		class _CDeviceArray1D
		{
		public:
			using TValue = _TValue;
			using TValuePtr = _TValue*;

		protected:
			TValuePtr m_pData;
			size_t m_nElementCount;
			size_t m_nElementCapacity;

		public:
			__CUDA_HDI__ TValuePtr DataPointer()
			{
				return m_pData;
			}

			__CUDA_HDI__ size_t ElementCount()
			{
				return m_nElementCount;
			}

			__CUDA_HDI__ size_t ElementCapacity()
			{
				return m_nElementCapacity;
			}

			__CUDA_HDI__ size_t ByteCount()
			{
				return ElementCount() * sizeof(TValue);
			}

			__CUDA_HDI__ size_t ByteCapacity()
			{
				return ElementCapacity() * sizeof(TValue);
			}

			bool IsValid() const
			{
				return (m_pData != nullptr) && (m_nElementCount > 0) && (m_nElementCapacity >= m_nElementCount);
			}

			void Clear()
			{
				m_pData = nullptr;
				m_nElementCapacity = 0;
				m_nElementCount = 0;
			}

			void Create(size_t nElCnt)
			{
				try
				{
					// If the current memory block is NOT valid or the required size
					// is larger than the current capacity...
					if (!IsValid() || nElCnt > m_nElementCapacity)
					{
						// ... then destroy the memory block and allocate one of the correct size.
						Destroy();
						Cuda::Malloc(&m_pData, nElCnt);
						m_nElementCount = nElCnt;
						m_nElementCapacity = nElCnt;
					}
					else
					{
						// ... otherwise only set the element count to the new value and keep the whole memory block.
						m_nElementCount = nElCnt;
					}

				}
				CLU_CATCH_RETHROW_ALL("Error creating 1D CUDA array")
			}

			void Destroy()
			{
				try
				{
					if (m_pData != nullptr)
					{
						Cuda::Free(m_pData);
					}
					Clear();
				}
				CLU_CATCH_RETHROW_ALL("Error destroying 1D CUDA array")
			}

			void ToDevice(const TValuePtr pData)
			{
				try
				{
					if (pData == nullptr)
					{
						throw CLU_EXCEPTION("Invalid data pointer given");
					}

					Cuda::MemCpy(m_pData, pData, m_nElementCount, ECopyType::HostToDevice);
				}
				CLU_CATCH_RETHROW_ALL("Error copying 1D array to device")
			}

			void ToDevice(const TValuePtr pData, size_t nByteOffset, size_t nByteCount)
			{
				try
				{
					if (pData == nullptr)
					{
						throw CLU_EXCEPTION("Invalid data pointer given");
					}

					if (nByteOffset + nByteCount > m_nElementCount)
					{
						throw CLU_EXCEPTION("Invalid device memory range");
					}

					Cuda::MemCpy(&m_pData[nByteOffset], pData, nByteCount, ECopyType::HostToDevice);
				}
				CLU_CATCH_RETHROW_ALL("Error copying 1D array to device")
			}


			void ToHost(TValuePtr pData) const
			{
				try
				{
					if (pData == nullptr)
					{
						throw CLU_EXCEPTION("Invalid data pointer given");
					}

					Cuda::MemCpy(pData, m_pData, m_nElementCount, ECopyType::DeviceToHost);
				}
				CLU_CATCH_RETHROW_ALL("Error copying 1D array to host")
			}

			void ToHost(TValuePtr pData, size_t nByteOffset, size_t nByteCount) const
			{
				try
				{
					if (pData == nullptr)
					{
						throw CLU_EXCEPTION("Invalid data pointer given");
					}

					if (nByteOffset + nByteCount > m_nElementCount)
					{
						throw CLU_EXCEPTION("Invalid device memory range");
					}

					Cuda::MemCpy(pData, &m_pData[nByteOffset], nByteCount, ECopyType::DeviceToHost);
				}
				CLU_CATCH_RETHROW_ALL("Error copying 1D array to host")
			}
		};



		template<typename _TValue>
		class CDeviceArray1D : public _CDeviceArray1D<_TValue>
		{
		public:
			CDeviceArray1D()
			{
				_CDeviceArray1D::Clear();
			}

			CDeviceArray1D(CDeviceArray1D&& xDevArray)
			{
				*this = std::forward<CDeviceArray1D>(xDevArray);
			}

			CDeviceArray1D& operator= (CDeviceArray1D&& xDevArray)
			{
				m_pData = xDevArray.m_pData;
				m_nElementCount = xDevArray.m_nElementCount;
				m_nElementCapacity = xDevArray.m_nElementCapacity;

				xDevArray.Clear();
				return *this;
			}


			CDeviceArray1D(const CDeviceArray1D& xDevArray) = delete;
			CDeviceArray1D& operator= (const CDeviceArray1D& xDevArray) = delete;

			~CDeviceArray1D()
			{
				Destroy();
			}


		};

	} // namespace Cuda
} // namespace Clu


////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Device.cpp
//
// summary:   Implements the device class
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

#include "Device.h"
#include "CluTec.Base/Exception.h"

namespace Clu
{
	namespace Cuda
	{


		CDevice::CDevice()
		{
			try
			{
				Clear();
			}
			CLU_CATCH_RETHROW_ALL("Error constructing device object");
		}

		CDevice::CDevice(int iIdx)
		{
			try
			{
				Set(iIdx);
			}
			CLU_CATCH_RETHROW_ALL("Error constructing device object");
		}


		CDevice::~CDevice()
		{
			Clear();
		}


		void CDevice::Set(int iIdx)
		{
			try
			{

				if (iIdx < 0)
				{
					throw CLU_EXCEPTION("Invalid device index");
				}

				if (iIdx >= Cuda::GetDeviceCount())
				{
					throw CLU_EXCEPTION("Device index out of range");
				}

				m_iDeviceIndex = iIdx;
				m_xProp = Cuda::GetDeviceProperties(iIdx);
			}
			CLU_CATCH_RETHROW_ALL("Error setting device index");
		}

		const cudaDeviceProp& CDevice::Properties() const
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Invalid device");
				}
				return m_xProp;
			}
			CLU_CATCH_RETHROW_ALL("Error obtaining device properties");
		}

		void CDevice::MakeCurrent() const
		{
			try
			{
				if (!IsValid())
				{
					throw CLU_EXCEPTION("Invalid device");
				}

				Cuda::SetDevice(m_iDeviceIndex);
			}
			CLU_CATCH_RETHROW_ALL("Error making device current");
		}


		bool CDevice::Supports(EDeviceProperty ePropId) const
		{
			try
			{
				bool bResult = false;

				switch (ePropId)
				{
				case EDeviceProperty::ManagedMemory:
					bResult = (m_xProp.managedMemory != 0);
					break;

				default:
					throw CLU_EXCEPTION("Given property is unkown or does not evaluate to a boolean value");
				}

				return bResult;
			}
			CLU_CATCH_RETHROW_ALL("Error obtaining boolean property value");
		}

		CDevice::TNumber CDevice::NumberOf(EDeviceProperty ePropId) const
		{
			try
			{
				int iResult = 0;

				switch (ePropId)
				{
				case EDeviceProperty::RegistersPerBlock:
					iResult = m_xProp.regsPerBlock;
					break;

				case EDeviceProperty::RegistersPerMultiprocessor:
					iResult = m_xProp.regsPerMultiprocessor;
					break;

				case EDeviceProperty::ThreadsPerBlock:
					iResult = m_xProp.maxThreadsPerBlock;
					break;

				default:
					throw CLU_EXCEPTION("Given property is unkown or does not evaluate to a positive number value");
				}

				return (TNumber)(iResult < 0 ? 0 : iResult);
			}
			CLU_CATCH_RETHROW_ALL("Error obtaining number property value");
		}

	}
}

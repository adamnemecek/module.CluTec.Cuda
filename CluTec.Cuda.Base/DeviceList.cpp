////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceList.cpp
//
// summary:   Implements the device list class
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

#include "CluTec.Base/Exception.h"

#include "DeviceList.h"

namespace Clu
{
	namespace Cuda
	{
		CDeviceList::TRef CDeviceList::m_pThis;

		CDeviceList::TRef CDeviceList::GetList()
		{
			if (m_pThis.use_count() == 0)
			{
				m_pThis = std::shared_ptr<CDeviceList>(new CDeviceList());
			}
			return m_pThis;
		}

		CDeviceList::CDeviceList()
		{
			_Init();
		}


		CDeviceList::~CDeviceList()
		{
		}


		void CDeviceList::_Init()
		{
			try
			{
				m_vecDevices.clear();

				int iCnt = (int)Cuda::GetDeviceCount();
				for (int iIdx = 0; iIdx < iCnt; ++iIdx)
				{
					m_vecDevices.emplace_back(CDevice(iIdx));
				}
			}
			CLU_CATCH_RETHROW_ALL("Error initializing device list")
		}

		const CDevice& CDeviceList::GetDevice(TSize nIdx)
		{
			try
			{
				if (nIdx >= (TSize)m_vecDevices.size())
				{
					throw CLU_EXCEPTION("Device index out of range");
				}

				return m_vecDevices[nIdx];
			}
			CLU_CATCH_RETHROW_ALL("Error obtaining device")
		}

	}
}

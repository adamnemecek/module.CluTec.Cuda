////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceEvent.cpp
//
// summary:   Implements the device event class
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
#include "CluTec.Base/Logger.h"

#include "DeviceEvent.h"


namespace Clu
{
	namespace Cuda
	{


		CDeviceEvent::CDeviceEvent()
		{
			try
			{
				Cuda::EventCreate(m_xEvent);
				m_xStream = nullptr;
			}
			CLU_CATCH_RETHROW_ALL("Error in device event constructor");
		}


		CDeviceEvent::~CDeviceEvent()
		{
			try
			{
				Cuda::EventDestroy(m_xEvent);
				m_xStream = nullptr;
			}
			CLU_LOG_DTOR_CATCH_ALL(CDeviceEvent)
		}

		void CDeviceEvent::SetStream(cudaStream_t xStream)
		{
			m_xStream = xStream;
		}

		void CDeviceEvent::Record()
		{
			try
			{
				Cuda::EventRecord(m_xEvent, m_xStream);
			}
			CLU_CATCH_RETHROW_ALL("Error recording event");
		}

		void CDeviceEvent::Sync()
		{
			try
			{
				Cuda::EventSynchronize(m_xEvent);
			}
			CLU_CATCH_RETHROW_ALL("Error in synchronizing event");
		}

		void CDeviceEvent::RecordAndSync()
		{
			Record();
			Sync();
		}


	}
}

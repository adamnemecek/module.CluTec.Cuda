////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      DeviceList.h
//
// summary:   Declares the device list class
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

#include <vector>
#include <memory>

#include "Api.h"
#include "Device.h"

namespace Clu
{
	namespace Cuda
	{

		class CDeviceList
		{
		public:
			using TSize = unsigned;

		public:
			using TRef = std::shared_ptr<CDeviceList>;

		private:
			static TRef m_pThis;

			std::vector<CDevice> m_vecDevices;

		private:
			void _Init();

		private:
			CDeviceList();

		public:
			~CDeviceList();

			static TRef GetList();

			TSize Count()
			{
				return (TSize)m_vecDevices.size();
			}

			const CDevice& GetDevice(TSize nIdx);

		};

	}
}

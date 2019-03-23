////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      MapDisparityTo3D.h
//
// summary:   Declares the map disparity to 3D class
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

#include "cuda_runtime.h"
#include "CluTec.Cuda.Base/Conversion.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.Base/KernelDriverBase.h"
#include "CluTec.ImgProc/Camera.StereoPinhole.h"

#include "CluTec.ImgProc/DisparityConfig.h"
#include "DisparityId.h"


namespace Clu
{
	namespace Cuda
	{
		namespace MapDisparityTo3D
		{
			using TValue = float;
			using TPixelMap3D = Clu::TPixel_RGBA_Single;
			using TPixelDisp = _SDisparityConfig::TPixel;
			using TPixelDispEx = _SDisparityConfig::TPixelEx;
			using TPixelDispF = _SDisparityConfig::TPixelF;
			using TDisp = _SDisparityConfig::TDisp;
			using TStereoPinhole = Clu::Camera::_CStereoPinhole<TValue>;

			struct _SParameter
			{
				TDisp uDispMin, uDispMax;

				TStereoPinhole camStereoPinhole;

				// Calculated Values
				_SDisparityConfig _xDispConfig;
				TValue _fScaleX, _fScaleY, _fScaleZ;
			};

			struct SParameter : public _SParameter
			{
				TPixelMap3D pixMin;
				TPixelMap3D pixMax;

				// If true, the output coordinate system is the FIP standard 
				// X to the Right, Y Up, Z Towards.
				// If false, the output coordinate system is like OpenCV or Halcon
				// X to the Right, Y Down, Z Away.
				bool bIsCsRUT;

				SParameter()
				{
				}

				template<typename TValue2>
				void Set(TDisp _uDispMin, TDisp _uDispMax, bool _bIsCsRUT, const Clu::Camera::_CStereoPinhole<TValue2>& _camStereoPinhole
						, int iMipLevelWidth, int iMipLevel)
				{
					uDispMin = _uDispMin;
					uDispMax = _uDispMax;
					bIsCsRUT = _bIsCsRUT;

					camStereoPinhole.CastFrom(_camStereoPinhole);
					_xDispConfig = camStereoPinhole.CreateDisparityConfig(iMipLevelWidth, iMipLevel);

					if (bIsCsRUT)
					{
						_fScaleX = TValue(1);
						_fScaleY = TValue(1);
						_fScaleZ = TValue(1);
					}
					else
					{
						_fScaleX = TValue(1);
						_fScaleY = TValue(-1);
						_fScaleZ = TValue(-1);
					}
				}

				void ResetRange()
				{
					pixMin.r() = Clu::NumericLimits<TValue>::Max();
					pixMin.g() = Clu::NumericLimits<TValue>::Max();
					pixMin.b() = Clu::NumericLimits<TValue>::Max();
					pixMin.a() = TValue(0);

					pixMax.r() = -Clu::NumericLimits<TValue>::Max();
					pixMax.g() = -Clu::NumericLimits<TValue>::Max();
					pixMax.b() = -Clu::NumericLimits<TValue>::Max();
					pixMax.a() = TValue(0);
				}
			};



			class CDriver : public Clu::Cuda::CKernelDriverBase
			{
			public:
#ifdef _DEBUG
				static const size_t NumberOfRegisters = 17;
#else
				static const size_t NumberOfRegisters = 17;
#endif
			private:
				SParameter m_xPars;
				Clu::CIImage m_imgRange;
				Clu::Cuda::CDeviceImage m_deviRange;

			public:
				CDriver()
					: CKernelDriverBase("Map disparity to 3D")
				{}

				~CDriver()
				{}

				const TPixelMap3D& RangeMin()
				{
					return m_xPars.pixMin;
				}

				const TPixelMap3D& RangeMax()
				{
					return m_xPars.pixMax;
				}

				void Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat, const SParameter& xPars);

				void Process(Clu::Cuda::_CDeviceSurface& surfMap3D, const Clu::Cuda::_CDeviceSurface& surfDisparity);
			};

		} // ImgProc
	} // Cuda
} // Clu


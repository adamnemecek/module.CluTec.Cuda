////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.UserTest1
// file:      UsertTest1.cpp
//
// summary:   Implements the usert test 1 class
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

#include "stdafx.h"

#include "CluTec.Types1/IImage.h"

#include "CluTec.Base/Logger.h"

#include "CluTec.Viz.View/Clu.Viz.View.h"

#include "CluTec.Cuda.Base/Device.h"
#include "CluTec.Cuda.Base/DeviceSurface.h"
#include "CluTec.Cuda.ImgBase/ConvertImageType.h"
#include "CluTec.Cuda.ImgProc/Statistics.DevFromMean.h"

int _tmain(int argc, _TCHAR* argv[])
{
	Clu::Cuda::CDevice xDevice;
	Clu::Cuda::Statistics::DevFromMean::CDriver xDevFromMean;
	Clu::Cuda::ConvertImageType::CDriver xConvertImageType;

	try
	{

		xDevice.Set(0);
		xDevice.MakeCurrent();

		Clu::Viz::View::Start();

		// ///////////////////////////////////////////////////////////////////////////////////////
		// ///////////////////////////////////////////////////////////////////////////////////////
		// ///////////////////////////////////////////////////////////////////////////////////////
		double dTime = 0.0;

		Clu::CIString sFileDir = R"(X:\CluTec\Solutions\CluTec.Sln.Halcon.1.0\CluTec.Halcon.1.0\CluTec.Halcon.v12\examples\images\Jadeplant\)";
		Clu::CIString sFileL = "im0_gray.png";
		Clu::CIString sFileR = "im1_gray.png";

		Clu::CIImage xImage = Clu::Viz::View::LoadImage(sFileDir + sFileL);
		//Clu::CIImage xImageR = Clu::Viz::View::LoadImage(R"(X:\_Images\im1.jpg)");
		//Clu::CIImage xImageDisp;

		Clu::SImageFormat xSrcF = xImage.Format();
		Clu::SImageFormat xTrgF(xSrcF);

		xTrgF.eDataType = Clu::EDataType::UInt8;
		xTrgF.ePixelType = Clu::EPixelType::Lum;

		Clu::Cuda::CDeviceImage deviA;
		Clu::Cuda::CDeviceSurface surfA, surfB;

		deviA.CopyFrom(xImage);
		surfA.Create(xTrgF);

		xConvertImageType.Configure(xDevice, xSrcF);
		dTime = Clu::Cuda::ProcessKernel(xConvertImageType, surfA, deviA);

		surfB.Create(surfA.Format());

		Clu::Cuda::Statistics::DevFromMean::SParameter xPars;
		xPars.Set(1.0f, 1.0f, 0.5f, 0.5f, Clu::Cuda::Statistics::DevFromMean::EConfig::Patch_16x16);

		xDevFromMean.Configure(xDevice, surfA.Format(), xPars);
		dTime = Clu::Cuda::ProcessKernel(xDevFromMean, surfB, surfA);

		Clu::CIImage xImageResult;
		surfB.CopyInto(xImageResult);

		Clu::Viz::View::SaveImage(R"(X:\_Images\result.bmp)", xImageResult);

		// ///////////////////////////////////////////////////////////////////////////////////////
		// ///////////////////////////////////////////////////////////////////////////////////////
		// ///////////////////////////////////////////////////////////////////////////////////////

		Clu::Viz::View::End();
	}
	catch (Clu::CIException& xEx)
	{
		printf("Error:\n%s\n", xEx.ToStringComplete().ToCString());
	}
	return 0;
}


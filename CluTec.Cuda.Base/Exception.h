////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Exception.h
//
// summary:   Declares the exception class
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

#include <string>

#include "cuda_runtime.h"

#include "CluTec.Types1/ExceptionTypes.h"
#include "CluTec.Types1/IException.h"
#include "CluTec.Base/Conversion.h"

#include "Api.h"

namespace Clu
{
	namespace ExceptionType
	{
		class Cuda : public Clu::ExceptionType::Unknown
		{
		public:
			// {FFF9A736-C596-4CD5-B3DB-FD306CFB3FF4}
			static const CGuid Guid;
			static const char* TypeName;

		public:
			Cuda() : Unknown(Cuda::Guid, Cuda::TypeName)
			{}

		};
	}


#define CLU_EXCEPT_CUDA(theErrorId, theFuncCall) \
	Clu::CreateExceptionCuda(theErrorId, theFuncCall, __FILE__, __FUNCTION__, __LINE__)

#define CLU_EXCEPT_CUDA_NEST(theErrorId, theFuncCall, theEx) \
	Clu::CreateExceptionCuda(theErrorId, theFuncCall, __FILE__, __FUNCTION__, __LINE__, theEx)

	template<typename T>
	inline CIException CreateExceptionCuda(cudaError_t uErrorId, const T& xMsg, const char* pcFile, const char* pcFunc, const int & iLine)
	{
		std::string sText = Clu::Cuda::GetErrorText(uErrorId, Clu::ToStdString(xMsg));

		return CIException(Clu::ExceptionType::Cuda(), Clu::ToIString(sText), Clu::ToIString(pcFile), Clu::ToIString(pcFunc), iLine);
	}

	template<typename T>
	inline CIException CreateExceptionCuda(cudaError_t uErrorId, const T& xMsg, const char* pcFile, const char* pcFunc, const int & iLine, CIException&& xEx)
	{
		std::string sText = Clu::Cuda::GetErrorText(uErrorId, ToStdString(xMsg));

		return CIException(Clu::ExceptionType::Cuda(), ToIString(sText), ToIString(pcFile), ToIString(pcFunc), iLine, std::forward<CIException>(xEx));
	}

}

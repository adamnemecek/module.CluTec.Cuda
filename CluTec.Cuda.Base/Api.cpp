////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Api.cpp
//
// summary:   Implements the API class
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

#include <cinttypes>
#include <sstream>

#include "CluTec.Base/Exception.h"

#include "Api.h"
#include "Exception.h"



namespace Clu
{

	namespace Cuda
	{


		std::string GetErrorText(cudaError_t uErrorId, const std::string& sFuncCall)
		{
			std::stringstream sxText;

			if (uErrorId != ::cudaSuccess)
			{
				sxText
					<< "Cuda error ("
					<< static_cast<uint32_t>(uErrorId)
					<< "): "
					<< cudaGetErrorString(uErrorId)
					<< " during call: "
					<< sFuncCall;
			}
			else
			{
				sxText << "No error during call: " << sFuncCall;
			}

			return sxText.str();
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \brief Cuda error initialise.
		///
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		void ResetError()
		{
			cudaGetLastError();
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \brief Cuda assert ok.
		///
		/// \param	uErrorId   Identifier for the error.
		/// \param	pcFuncCall The PC function call.
		/// \param	pcFile	   The PC file.
		/// \param	iLine	   Zero-based index of the line.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		void EnsureOK(cudaError_t uErrorId, const std::string& sFuncCall, const char* const pcFile, const char* const pcFunction, int const iLine)
		{
			if (uErrorId != ::cudaSuccess)
			{
				ResetError();
				throw CreateExceptionCuda(uErrorId, sFuncCall, pcFile, pcFunction, iLine);
			}

		}


		size_t GetDeviceCount()
		{
			int iDevCnt = 0;
			CLU_CUDA_ENSURE_OK(cudaGetDeviceCount(&iDevCnt));

			return (size_t)(iDevCnt < 0 ? 0 : iDevCnt);
		}

		cudaDeviceProp GetDeviceProperties(int iDevIdx)
		{
			cudaDeviceProp xDevProp;
			CLU_CUDA_ENSURE_OK(cudaGetDeviceProperties(&xDevProp, iDevIdx));
			return xDevProp;
		}

		void SetDevice(int iDevIdx)
		{
			CLU_CUDA_ENSURE_OK(cudaSetDevice(iDevIdx));
		}


		void DeviceSynchronize()
		{
			CLU_CUDA_ENSURE_OK(cudaDeviceSynchronize());
		}


		template<> void Malloc(void** pvData, size_t nByteCnt)
		{
			CLU_CUDA_ENSURE_OK(cudaMalloc((void **)pvData, nByteCnt));
		}

		template<> void MemCpy(void* pDest, const void* pSrc, size_t nByteCnt, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpy((void *)pDest, (const void*)pSrc, nByteCnt, cudaMemcpyKind(eType)));
		}


		void EventCreate(cudaEvent_t &xEvent)
		{
			CLU_CUDA_ENSURE_OK(cudaEventCreate(&xEvent));
		}
		void EventDestroy(cudaEvent_t& xEvent)
		{
			CLU_CUDA_ENSURE_OK(cudaEventDestroy(xEvent));
		}

		void EventRecord(cudaEvent_t &xEvent, cudaStream_t& xStream)
		{
			CLU_CUDA_ENSURE_OK(cudaEventRecord(xEvent, xStream));
		}

		void EventSynchronize(cudaEvent_t& xEvent)
		{
			CLU_CUDA_ENSURE_OK(cudaEventSynchronize(xEvent));
		}

		double EventElapsedTime(cudaEvent_t& evStart, cudaEvent_t& evStop)
		{
			float fTime = 0.0f;
			CLU_CUDA_ENSURE_OK(cudaEventElapsedTime(&fTime, evStart, evStop));
			return (double)fTime;
		}


		void MallocArray(cudaArray_t* ppArray, const cudaChannelFormatDesc *pCfd, size_t nWidth, size_t nHeight, 
			EDeviceArrayAllocation eAlloc)
		{
			CLU_CUDA_ENSURE_OK(cudaMallocArray(ppArray, pCfd, nWidth, nHeight, (unsigned int)eAlloc));
		}

		void FreeArray(cudaArray_t pArray)
		{
			CLU_CUDA_ENSURE_OK(cudaFreeArray(pArray));
		}

		void MemCpyToArray(cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY, const void *pData, size_t nByteCount, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpyToArray(pArray, nOffsetX, nOffsetY, pData, nByteCount, cudaMemcpyKind(eType)));
		}

		void MemCpyFromArray(void *pData, cudaArray_const_t pArray, size_t nOffsetX, size_t nOffsetY, size_t nByteCount, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpyFromArray(pData, pArray, nOffsetX, nOffsetY, nByteCount, cudaMemcpyKind(eType)));
		}

		void MemCpy2DToArray(cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY, const void *pData
			, size_t nSrcPitch, size_t nSrcWidth, size_t nSrcHeight, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpy2DToArray(pArray, nOffsetX, nOffsetY, pData, nSrcPitch, nSrcWidth, nSrcHeight, cudaMemcpyKind(eType)));
		}

		void MemCpy2DFromArray(void *pData, size_t nTrgPitch, cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY
			, size_t nSrcWidth, size_t nSrcHeight, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpy2DFromArray(pData, nTrgPitch, pArray, nOffsetX, nOffsetY, nSrcWidth, nSrcHeight, cudaMemcpyKind(eType)));
		}


		void MemCpy2DArrayToArray(cudaArray_t pArrayTrg, size_t nTrgOrigX, size_t nTrgOrigY
			, cudaArray_const_t pArraySrc, size_t nSrcOrigX, size_t nSrcOrigY
			, size_t nSrcW, size_t nSrcH)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpy2DArrayToArray(pArrayTrg, nTrgOrigX, nTrgOrigY, pArraySrc, nSrcOrigX, nSrcOrigY, nSrcW, nSrcH, cudaMemcpyDeviceToDevice));
		}


		void CreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc)
		{
			CLU_CUDA_ENSURE_OK(cudaCreateSurfaceObject(pSurfObject, pResDesc));
		}

		void DestroySurfaceObject(cudaSurfaceObject_t surfObject)
		{
			CLU_CUDA_ENSURE_OK(cudaDestroySurfaceObject(surfObject));
		}

		void CreateTextureObject(cudaTextureObject_t *pTexObject, const cudaResourceDesc* pResDesc
			, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc)
		{
			CLU_CUDA_ENSURE_OK(cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
		}

		void DestroyTextureObject(cudaTextureObject_t texObject)
		{
			CLU_CUDA_ENSURE_OK(cudaDestroyTextureObject(texObject));
		}


		void GraphicsGLRegisterBuffer(cudaGraphicsResource_t* ppResource, unsigned uOpenGLBufferId, EMapGraphicsType eMapType)
		{
			CLU_CUDA_ENSURE_OK(cudaGraphicsGLRegisterBuffer(ppResource, uOpenGLBufferId, (unsigned)eMapType));
		}

		void GraphicsUnregisterResource(cudaGraphicsResource_t pResource)
		{
			CLU_CUDA_ENSURE_OK(cudaGraphicsUnregisterResource(pResource));
		}

		void GraphicsMapResources(int iCount, cudaGraphicsResource_t* ppResources, cudaStream_t xStream)
		{
			CLU_CUDA_ENSURE_OK(cudaGraphicsMapResources(iCount, ppResources, xStream));
		}

		void GraphicsUnmapResources(int iCount, cudaGraphicsResource_t* ppResources, cudaStream_t xStream)
		{
			CLU_CUDA_ENSURE_OK(cudaGraphicsUnmapResources(iCount, ppResources, xStream));
		}

		void GraphicsResourceGetMappedPointer(void **ppData, size_t &nSize, cudaGraphicsResource_t pResource)
		{
			CLU_CUDA_ENSURE_OK(cudaGraphicsResourceGetMappedPointer(ppData, &nSize, pResource));
		}



	}

}


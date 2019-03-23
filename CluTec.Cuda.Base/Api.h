////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      Api.h
//
// summary:   Declares the API class
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

#if (defined(_M_IX86) || defined(_M_IA64) || defined(_M_AMD64) || defined(_M_ARM)) && !defined(MIDL_PASS)
#define DECLSPEC_IMPORT __declspec(dllimport)
#else
#define DECLSPEC_IMPORT
#endif

#if !defined(_GDI32_)
#define WINGDIAPI DECLSPEC_IMPORT
#else
#define WINGDIAPI
#endif

#if !defined(APIENTRY)
#	define APIENTRY
#endif

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#define CLU_CUDA_ENSURE_OK(theValue) Clu::Cuda::EnsureOK((theValue), #theValue, __FILE__, __FUNCTION__, __LINE__)

 
namespace Clu
{
	namespace Cuda
	{
		/// \brief Values that represent memories.
		enum class EMemory
		{
			AttachGlobal = cudaMemAttachGlobal,
			AttachHost = cudaMemAttachHost,
		};

		enum class ECopyType
		{
			HostToDevice = cudaMemcpyHostToDevice,
			DeviceToHost = cudaMemcpyDeviceToHost,
			DeviceToDevice = cudaMemcpyDeviceToDevice,
		};

		enum class EDeviceArrayAllocation
		{
			Default = cudaArrayDefault,
			SurfaceLoadStore = cudaArraySurfaceLoadStore,
		};

		enum class ETextureAddressMode
		{
			Wrap = cudaAddressModeWrap,
			Clamp = cudaAddressModeClamp,
			Mirror = cudaAddressModeMirror,
			Border = cudaAddressModeBorder,
		};

		enum class ETextureFilterMode
		{
			Point = cudaFilterModePoint,
			Linear = cudaFilterModeLinear,
		};

		enum class ETextureReadMode
		{
			ElementType = cudaReadModeElementType,
			NormalizedFloat = cudaReadModeNormalizedFloat,
		};

		enum class EMapGraphicsType
		{
			/// <summary>	Default; Assume resource can be read / written. </summary>
			ReadWrite = cudaGraphicsMapFlagsNone,
			/// <summary>	CUDA will not write to this resource. </summary>
			ReadOnly = cudaGraphicsMapFlagsReadOnly,
			/// <summary>	CUDA will only write to and will not read from this resource. </summary>
			WriteOnly = cudaGraphicsMapFlagsWriteDiscard,
		};


		std::string GetErrorText(cudaError_t uErrorId, const std::string& sFuncCall);

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \brief Cuda error initialise.
		///
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		void ResetError();

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/// \brief Cuda assert ok.
		///
		/// \param	uErrorId   Identifier for the error.
		/// \param	pcFuncCall The PC function call.
		/// \param	pcFile	   The PC file.
		/// \param	iLine	   Zero-based index of the line.
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		void EnsureOK(cudaError_t uErrorId, const std::string& pcFuncCall, const char* const pcFile, const char* const pcFunction, int const iLine);


		size_t GetDeviceCount();
		cudaDeviceProp GetDeviceProperties(int iDevIdx);
		void SetDevice(int iDevIdx);

		void DeviceSynchronize();

		void EventCreate(cudaEvent_t &xEvent);
		void EventDestroy(cudaEvent_t &xEvent);
		void EventRecord(cudaEvent_t& xEvent, cudaStream_t& xStream);
		void EventSynchronize(cudaEvent_t& xEvent);
		double EventElapsedTime(cudaEvent_t& evStart, cudaEvent_t& evStop);

		void MallocArray(cudaArray_t* ppArray, const cudaChannelFormatDesc *pCfd, size_t nWidth, size_t nHeight, 
			EDeviceArrayAllocation eAlloc);

		void FreeArray(cudaArray_t pArray);

		void MemCpyToArray(cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY, const void *pData, size_t nByteCount, ECopyType eType);
		void MemCpyFromArray(void *pData, cudaArray_const_t pArray, size_t nOffsetX, size_t nOffsetY, size_t nByteCount, ECopyType eType);

		void MemCpy2DToArray(cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY, const void *pData
							, size_t nSrcPitch, size_t nSrcWidth, size_t nSrcHeight, ECopyType eType);

		void MemCpy2DFromArray(void *pData, size_t nTrgPitch, cudaArray_t pArray, size_t nOffsetX, size_t nOffsetY
							, size_t nSrcWidth, size_t nSrcHeight, ECopyType eType);

		void MemCpy2DArrayToArray(cudaArray_t pArrayTrg, size_t nTrgOrigX, size_t nTrgOrigY
			, cudaArray_const_t pArraySrc, size_t nSrcOrigX, size_t nSrcOrigY
			, size_t nSrcW, size_t nSrcH);

#pragma region Template Functions
		template<typename TValue>
		void MallocManaged(TValue **pData, size_t nElCnt, EMemory eFlags = EMemory::AttachGlobal)
		{
			CLU_CUDA_ENSURE_OK(cudaMallocManaged(pData, nElCnt * sizeof(TValue), static_cast<unsigned int>(eFlags)));
		}

		template<typename TValue>
		void Malloc(TValue** pData, size_t nElCnt)
		{
			CLU_CUDA_ENSURE_OK(cudaMalloc((void **)pData, nElCnt * sizeof(TValue)));
		}

		template<> void Malloc(void** pvData, size_t nByteCnt);

		template<typename TValue>
		void Free(TValue* pData)
		{
			if (pData == nullptr)
			{
				return;
			}

			CLU_CUDA_ENSURE_OK(cudaFree((void*)pData));
		}

		//template<typename TValue>
		//void MemCpy(TValue* pDest, const TValue* pSrc, size_t nElCnt, ECopyType eType)
		//{
		//	CLU_CUDA_ENSURE_OK(cudaMemcpy((void *)pDest, (const void*)pSrc, nElCnt * sizeof(TValue), cudaMemcpyKind(eType)));
		//}


		template<typename TValue1, typename TValue2>
		void MemCpy(TValue1* pDest, const TValue2* pSrc, size_t nElCnt, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpy((void *)pDest, (const void*)static_cast<const TValue1*>(pSrc), nElCnt * sizeof(TValue1), cudaMemcpyKind(eType)));
		}

		template<> void MemCpy(void* pDest, const void* pSrc, size_t nByteCnt, ECopyType eType);


		//template<typename TValue>
		//void MemCpyToSymbol(TValue* pDest, const TValue* pSrc, size_t nElCnt, ECopyType eType)
		//{
		//	CLU_CUDA_ENSURE_OK(cudaMemcpyToSymbol((void *)pDest, (const void*)pSrc, nElCnt * sizeof(TValue), cudaMemcpyKind(eType)));
		//}

		template<typename TValue1, typename TValue2>
		void MemCpyToSymbol(TValue1& xDest, const TValue2* pSrc, size_t nElCnt, size_t nElOffset, ECopyType eType)
		{
			CLU_CUDA_ENSURE_OK(cudaMemcpyToSymbol(xDest, (const void*)static_cast<const TValue1*>(pSrc), nElCnt * sizeof(TValue1), nElOffset * sizeof(TValue1), cudaMemcpyKind(eType)));
		}


#pragma endregion

		void CreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc);
		void DestroySurfaceObject(cudaSurfaceObject_t surfObject);

		void CreateTextureObject(cudaTextureObject_t *pTexObject, const cudaResourceDesc* pResDesc
			, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc);
		void DestroyTextureObject(cudaTextureObject_t texObject);


		void GraphicsGLRegisterBuffer(cudaGraphicsResource_t* ppResource, unsigned uOpenGLBufferId, EMapGraphicsType eMapType);
		void GraphicsUnregisterResource(cudaGraphicsResource_t pResource);

		void GraphicsMapResources(int iCount, cudaGraphicsResource_t* ppResources, cudaStream_t xStream = nullptr);
		void GraphicsUnmapResources(int iCount, cudaGraphicsResource_t* ppResources, cudaStream_t xStream = nullptr);

		void GraphicsResourceGetMappedPointer(void **ppData, size_t &nSize, cudaGraphicsResource_t pResource);

	}
}


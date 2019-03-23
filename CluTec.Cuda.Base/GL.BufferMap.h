////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.Base
// file:      GL.BufferMap.h
//
// summary:   Declares the gl. buffer map class
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
#include "CluTec.Base/Exception.h"
#include "CluTec.Base/Logger.h"

#include "Api.h"

namespace Clu
{
	namespace Cuda
	{
		namespace GL
		{

			template<typename _TElement>
			class CBufferMap
			{
			public:
				using TElement = _TElement;

			protected:
				unsigned m_uGLBufferId;
				cudaGraphicsResource_t m_pResource;
				TElement *m_pBoundData;

			protected:
				void _Reset()
				{
					m_uGLBufferId = 0;
					m_pResource = nullptr;
					m_pBoundData = nullptr;
				}

			public:
				CBufferMap()
				{
					_Reset();
				}

				~CBufferMap()
				{
					try
					{
						Destroy();
					}
					CLU_LOG_DTOR_CATCH_ALL(CBufferMap)
				}

				CBufferMap(const CBufferMap&) = delete;
				CBufferMap& operator= (const CBufferMap&) = delete;


				bool IsValid()
				{
					return (m_uGLBufferId != 0) && (m_pResource != nullptr);
				}

				bool IsBound()
				{
					return m_pBoundData != nullptr;
				}

				TElement* BoundDataPointer()
				{
					return m_pBoundData;
				}

				void Create(unsigned uGLBufferId, EMapGraphicsType eMapType)
				{
					try
					{
						Destroy();

						GraphicsGLRegisterBuffer(&m_pResource, uGLBufferId, eMapType);
						m_uGLBufferId = uGLBufferId;
					}
					CLU_CATCH_RETHROW_ALL("Error creating OpenGL buffer map")
				}

				void Destroy()
				{
					try
					{
						if (!IsValid())
						{
							return;
						}

						if (IsBound())
						{
							Unbind();
						}

						GraphicsUnregisterResource(m_pResource);

						_Reset();
					}
					CLU_CATCH_RETHROW_ALL("Error destroying OpenGL buffer map")
				}


				TElement* Bind()
				{
					try
					{
						if (!IsValid())
						{
							throw CLU_EXCEPTION("Invalid OpenGL buffer map");
						}

						if (IsBound())
						{
							return BoundDataPointer();
						}

						size_t nSize = 0;
						GraphicsMapResources(1, &m_pResource);
						GraphicsResourceGetMappedPointer((void **)&m_pBoundData, nSize, m_pResource);

						return m_pBoundData;
					}
					CLU_CATCH_RETHROW_ALL("Error binding OpenGL buffer map")
				}

				void Unbind()
				{
					try
					{
						if (!IsValid())
						{
							throw CLU_EXCEPTION("Invalid OpenGL buffer map");
						}

						if (!IsBound())
						{
							return;
						}

						GraphicsUnmapResources(1, &m_pResource);
						m_pBoundData = nullptr;
					}
					CLU_CATCH_RETHROW_ALL("Error un-binding OpenGL buffer map")
				}
			};
		} // GL
	} // Cuda
} // Clu


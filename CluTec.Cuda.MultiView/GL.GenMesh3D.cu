////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      GL.GenMesh3D.cu
//
// summary:   gl. generate mesh 3D class
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

#include "GL.GenMesh3D.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"
#include "CluTec.Math/Static.Vector.h"

#include "CluTec.Cuda.Base/PixelTypeInfo.h"
#include "CluTec.Cuda.Base/Conversion.h"

#include "CluTec.OpenGL/Vertex.Data.h"

namespace Clu
{
	namespace Cuda
	{
		namespace GL
		{
			namespace GenMesh3D
			{
				namespace Kernel
				{
					struct Const
					{
						static const int WarpsPerBlockX = 4;
						static const int WarpsPerBlockY = 1;
						static const int ThreadCountX = 8;
						static const int ThreadCountY = 16;
						static const int BlockSizeX = ThreadCountX;
						static const int BlockSizeY = ThreadCountY;
					};

					__constant__ Clu::Cuda::GL::GenMesh3D::_SParameter c_xPars;

					////////////////////////////////////////////////////////////////////////////////////////////////////
					/// <summary>	Maps a disparity image to a metric 3D data set. </summary>
					///
					/// <param name="surfMap3D">		[in,out] The surf map 3D. </param>
					/// <param name="surfDisparity">	The surf disparity. </param>
					////////////////////////////////////////////////////////////////////////////////////////////////////

					template<typename TVertex>
					__global__ void Generate(TVertex* pVexData, Clu::Cuda::_CDeviceSurface surfMap3D)
					{
						using TMap3D = typename Clu::Cuda::SPixelTypeInfo<TPixelMap3D>::TElement;
						using TFloat = typename TPixelMap3D::TData;
						using TVec3 = Clu::_SVector<TFloat, 3>;

						int iSrcX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
						int iSrcY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);

						if (!surfMap3D.IsInside(iSrcX, iSrcY))
						{
							return;
						}

						TVertex *pVexTrg = &(pVexData[iSrcY * c_xPars.iVexCountX + iSrcX]);
						
						// Make border pixel calculations equal to those with sufficient border.
						iSrcX = max(iSrcX, 1);
						iSrcX = min(iSrcX, surfMap3D.Format().iWidth - 2);
						iSrcY = max(iSrcY, 1);
						iSrcY = min(iSrcY, surfMap3D.Format().iHeight - 2);

						TFloat fVexValid = TFloat(0);
						TVec3 vC, vR, vT, vL, vB;

						// Read source pixel
						{
							TPixelMap3D pData[5];
							pData[0] = surfMap3D.ReadPixel2D<TPixelMap3D>(iSrcX, iSrcY);
							pData[1] = surfMap3D.ReadPixel2D<TPixelMap3D>(iSrcX + 1, iSrcY);
							pData[2] = surfMap3D.ReadPixel2D<TPixelMap3D>(iSrcX, iSrcY - 1);
							pData[3] = surfMap3D.ReadPixel2D<TPixelMap3D>(iSrcX - 1, iSrcY);
							pData[4] = surfMap3D.ReadPixel2D<TPixelMap3D>(iSrcX, iSrcY + 1);

							vC.SetElements(pData[0].r(), pData[0].g(), pData[0].b());
							vR.SetElements(pData[1].r(), pData[1].g(), pData[1].b());
							vT.SetElements(pData[2].r(), pData[2].g(), pData[2].b());
							vL.SetElements(pData[3].r(), pData[3].g(), pData[3].b());
							vB.SetElements(pData[4].r(), pData[4].g(), pData[4].b());

							fVexValid = pData[0].a();
						}

						// Write position vector
						pVexTrg->Position(0) = c_xPars._fScale * (vC[0] - c_xPars._vCenter[0]);
						pVexTrg->Position(1) = c_xPars._fScale * (vC[1] - c_xPars._vCenter[1]);
						pVexTrg->Position(2) = c_xPars._fScale * (vC[2] - c_xPars._vCenter[2]);

						// Calculate Normal
						{
							TVec3 pvN[4];
							
							vR -= vC;
							vT -= vC;
							vL -= vC;
							vB -= vC;

							// Calculate normals of 4 adjacent triangles
							pvN[0] = Clu::Normalize(vR ^ vT);
							pvN[1] = Clu::Normalize(vT ^ vL);
							pvN[2] = Clu::Normalize(vL ^ vB);
							pvN[3] = Clu::Normalize(vB ^ vR);

							// Mean normal
							pvN[0] += pvN[1];
							pvN[2] += pvN[3];
							pvN[0] += pvN[2];
							pvN[0] /= TFloat(4);

							// Write normal vector
							pVexTrg->Normal(0) = pvN[0][0];
							pVexTrg->Normal(1) = pvN[0][1];
							pVexTrg->Normal(2) = pvN[0][2];
						}

						// Calculate texture coordinate
						{
							TFloat fTexX, fTexY;

							fTexX = (TFloat(1) / TFloat(c_xPars.iVexCountX)) * (TFloat(iSrcX) + TFloat(0.5));
							fTexY = (TFloat(1) / TFloat(c_xPars.iVexCountY)) * (TFloat(c_xPars.iVexCountY - iSrcY - 1) + TFloat(0.5));

							pVexTrg->Texture(0) = fTexX;
							pVexTrg->Texture(1) = fTexY;
							pVexTrg->Texture(2) = fVexValid;
						}

					}

				} // namespace Kernel

				// //////////////////////////////////////////////////////////////////////////////////////////////////////////
				// ///////////////////////////////////////////////////////////////////////////////////////////////////////
				// //////////////////////////////////////////////////////////////////////////////////////////////////////////
				// DRIVER
				// //////////////////////////////////////////////////////////////////////////////////////////////////////////
				// //////////////////////////////////////////////////////////////////////////////////////////////////////////
				// //////////////////////////////////////////////////////////////////////////////////////////////////////////

				void CDriver::Configure(const Clu::Cuda::CDevice& xDevice, const Clu::SImageFormat& xFormat
					, const SParameter& xPars)
				{
					if (xPars.iVexCountX != xFormat.iWidth
						|| xPars.iVexCountY != xFormat.iHeight)
					{
						throw CLU_EXCEPTION("GL-Buffer dimension differs from depth map image format");
					}

					m_xPars = xPars;

					// Calculate parameter values needed by kernel
					using TFloat = TPixelMap3D::TData;
					using TVec3 = Clu::_SVector<TFloat, 3>;

					TFloat fMaxRange;
					TVec3 vMin, vMax, vRange;

					vMin.SetElements(m_xPars.pixMin.r(), m_xPars.pixMin.g(), m_xPars.pixMin.b());
					vMax.SetElements(m_xPars.pixMax.r(), m_xPars.pixMax.g(), m_xPars.pixMax.b());
					vRange = vMax - vMin;
					m_xPars._vCenter = vMin + vRange / TFloat(2);

					fMaxRange = vRange[0];
					fMaxRange = Clu::Max(fMaxRange, vRange[1]);
					fMaxRange = Clu::Max(fMaxRange, vRange[2]);

					m_xPars._fScale = TFloat(3) / fMaxRange;



					EvalThreadConfigBlockSize(xDevice, xFormat
						, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
						, 0, 0, 0, 0 // Offsets
						, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
						, NumberOfRegisters
						, false // Use also partial blocks
						);
				}

				template<typename TVertex>
				void CDriver::Process(Clu::Cuda::GL::CBufferMap<TVertex>& xGlBuffer, const Clu::Cuda::_CDeviceSurface& surfMap3D)
				{
					if (!surfMap3D.IsOfType<TPixelMap3D>())
					{
						throw CLU_EXCEPTION("Given 3D-map image is not of the correct type");
					}

					if (!xGlBuffer.IsValid())
					{
						throw CLU_EXCEPTION("Given OpenGL buffer map is invalid");
					}

					if (m_xPars.iVexCountX != surfMap3D.Format().iWidth
						|| m_xPars.iVexCountY != surfMap3D.Format().iHeight)
					{
						throw CLU_EXCEPTION("GL-Buffer dimension differs from depth map image format");
					}

					Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

					TVertex *pVexData = xGlBuffer.Bind();

					Kernel::Generate
						CLU_KERNEL_CONFIG()
						(pVexData, surfMap3D);

					xGlBuffer.Unbind();
				}


				using TVexStd = Clu::OpenGL::Vertex::_SData<Clu::OpenGL::Vertex::EType::Standard>;

				template void CDriver::Process<TVexStd>(
					Clu::Cuda::GL::CBufferMap<TVexStd>& xGlBuffer, const Clu::Cuda::_CDeviceSurface& surfMap3D);

			} // GenMesh3D
		} // GL
	} // Cuda
} // Clu


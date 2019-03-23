////////////////////////////////////////////////////////////////////////////////////////////////////
// project:   CluTec.Cuda.MultiView
// file:      MapDisparityTo3D.cu
//
// summary:   map disparity to 3D class
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

#include "MapDisparityTo3D.h"
#include "CluTec.Types1/Pixel.h"
#include "CluTec.Math/Conversion.h"
#include "CluTec.Math/Static.Geometry.h"
#include "CluTec.Math/Static.Geometry.Math.h"

#include "CluTec.Cuda.Base/PixelTypeInfo.h"
#include "CluTec.Cuda.Base/Conversion.h"

//#define CLU_DEBUG_KERNEL

#include "CluTec.Cuda.Base/Kernel.Debug.h"

//#define DEBUG_LOC 58, 0, 39, 16
#define DEBUG_LOC 0, 0, 5, 1


namespace Clu
{
	namespace Cuda
	{
		namespace MapDisparityTo3D
		{
			namespace Kernel
			{
				using namespace Clu::Cuda::Kernel;

				struct Const
				{
					static const int WarpsPerBlockX = 4;
					static const int WarpsPerBlockY = 1;
					static const int ThreadCountX = 8;
					static const int ThreadCountY = 16;
					static const int BlockSizeX = ThreadCountX;
					static const int BlockSizeY = ThreadCountY;
					static const int ThreadsPerBlockX = WarpsPerBlockX * 32;
				};

				__constant__ Clu::Cuda::MapDisparityTo3D::_SParameter c_xPars;

				////////////////////////////////////////////////////////////////////////////////////////////////////
				/// <summary>	Maps a disparity image to a metric 3D data set. </summary>
				///
				/// <param name="surfMap3D">		[in,out] The surf map 3D. </param>
				/// <param name="surfDisparity">	The surf disparity. </param>
				////////////////////////////////////////////////////////////////////////////////////////////////////

				template<typename TPixelDisp>
				__global__ void Map(Clu::Cuda::_CDeviceSurface surfMap3D, Clu::Cuda::_CDeviceImage deviRange
					, const Clu::Cuda::_CDeviceSurface surfDisparity)
				{
					using TMap3D = typename Clu::Cuda::SPixelTypeInfo<TPixelMap3D>::TElement;
					using TMap3DComp = typename TPixelMap3D::TData;

					using TDispTuple = typename Clu::Cuda::SPixelTypeInfo<TPixelDisp>::TElement;
					using TDisp = typename TPixelDisp::TData;

					using TVec3 = typename TStereoPinhole::TVec3;
					using TLine3D = Clu::SLine3D<TValue>;

					using TFloat = TPixelMap3D::TData;

					int iLeftX = int(blockIdx.x * Const::BlockSizeX + threadIdx.x % Const::BlockSizeX);
					int iLeftY = int(blockIdx.y * Const::BlockSizeY + threadIdx.x / Const::BlockSizeX);

					if (!surfDisparity.IsInside(iLeftX, iLeftY))
					{
						return;
					}

					TPixelMap3D pixMap3D;
					pixMap3D.r() = TFloat(0);
					pixMap3D.g() = TFloat(0);
					pixMap3D.b() = TFloat(0);
					pixMap3D.a() = TFloat(0);

					TPixelDisp pixDisp;
					pixDisp = surfDisparity.ReadPixel2D<TPixelDisp>(iLeftX, iLeftY);

					TDisp uDisp = pixDisp.r();



					//int iIsValid = 1;
					//uDisp = c_xPars.xDispConfig.OffsetLR() + c_xPars.xDispConfig.Range() / 2;

					int iIsValid = int(uDisp >= c_xPars.uDispMin && uDisp <= c_xPars.uDispMax);
					//int iIsValid = int(uDisp >= TDisp(EDisparityId::First) && uDisp <= TDisp(EDisparityId::Last));

					if (iIsValid)
					{
						uDisp -= TDisp(EDisparityId::First);
					}
					else
					{
						uDisp = TDisp(c_xPars.uDispMin);
					}

					float fDisp = float(uDisp);
					float fLeftX = float(iLeftX);
					float fLeftY = float(iLeftY);

					float fRightX = c_xPars._xDispConfig.MapPixelPosLeftToRight(fLeftX, fDisp);
					float fRightY = fLeftY;

					Debug::Run([&]()
					{
						//printf("%d, %d, %d, %d: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, uDisp);
						if (Debug::IsThreadAndBlock(DEBUG_LOC))
						{
							printf("Disp, iXL, iXR: %d, %g, %g\n", uDisp, fLeftX, fRightX);
						}
					});

					TVec3 vPosL, vPosR;

					c_xPars.camStereoPinhole[0].Map_PixelIX_to_WorldM(vPosL, fLeftX, fLeftY, surfDisparity.Format());
					c_xPars.camStereoPinhole[1].Map_PixelIX_to_WorldM(vPosR, fRightX, fRightY, surfDisparity.Format());

					Debug::Run([&]()
					{
						if (Debug::IsThreadAndBlock(DEBUG_LOC))
						{
							printf("vPosL/R: (%g, %g, %g) / (%g, %g, %g)\n", vPosL[0], vPosL[1], vPosL[2], vPosR[0], vPosR[1], vPosR[2]);
						}
					});

					TLine3D xProjLineLeft, xProjLineRight;

					xProjLineLeft.vOrigin = c_xPars.camStereoPinhole[0].PinholeM_w();
					xProjLineLeft.vDir = Normalize(vPosL - xProjLineLeft.vOrigin);

					xProjLineRight.vOrigin = c_xPars.camStereoPinhole[1].PinholeM_w();
					xProjLineRight.vDir = Normalize(vPosR - xProjLineRight.vOrigin);

					Debug::Run([&]()
					{
						if (Debug::IsThreadAndBlock(DEBUG_LOC))
						{
							printf("Line Left: (%g, %g, %g) / (%g, %g, %g)\n", xProjLineLeft.vOrigin[0], xProjLineLeft.vOrigin[1], xProjLineLeft.vOrigin[2]
								, xProjLineLeft.vDir[0], xProjLineLeft.vDir[1], xProjLineLeft.vDir[2]);
							printf("Line Right: (%g, %g, %g) / (%g, %g, %g)\n", xProjLineRight.vOrigin[0], xProjLineRight.vOrigin[1], xProjLineRight.vOrigin[2]
								, xProjLineRight.vDir[0], xProjLineRight.vDir[1], xProjLineRight.vDir[2]);
						}
					});

					bool bHasIntersection = false;
					TVec3 vResultPos;
					TFloat fR, fG, fB;

					if ((bHasIntersection = Clu::TryIntersect(vResultPos, xProjLineLeft, xProjLineRight, Clu::CValuePrecision<TValue>::DefaultPrecision())))
					{
						Debug::Run([&]()
						{
							if (Debug::IsThreadAndBlock(DEBUG_LOC))
							{
								printf("vResult: (%g, %g, %g)\n", vResultPos[0], vResultPos[1], vResultPos[2]);
							}
						});

						fR = TFloat(c_xPars._fScaleX) * TFloat(vResultPos[0]);
						fG = TFloat(c_xPars._fScaleY) * TFloat(vResultPos[1]);
						fB = TFloat(c_xPars._fScaleZ) * TFloat(vResultPos[2]);

						pixMap3D.r() = fR;
						pixMap3D.g() = fG;
						pixMap3D.b() = fB;
						pixMap3D.a() = TFloat(iIsValid);
					}


					int *pLock = (int*)(&deviRange.At<TPixelMap3D>(0, 0).a());
					if (threadIdx.x == 0)
					{
						while (atomicCAS(pLock, 0, 1));
					}

					__syncthreads();
					//threadfence();

					if (iIsValid && bHasIntersection)
					{

						TPixelMap3D pixMin = deviRange.At<TPixelMap3D>(threadIdx.x, 0);
						TPixelMap3D pixMax = deviRange.At<TPixelMap3D>(threadIdx.x, 1);

						pixMin.r() = min(pixMin.r(), fR);
						pixMin.g() = min(pixMin.g(), fG);
						pixMin.b() = min(pixMin.b(), fB);

						pixMax.r() = max(pixMax.r(), fR);
						pixMax.g() = max(pixMax.g(), fG);
						pixMax.b() = max(pixMax.b(), fB);

						deviRange.At<TPixelMap3D>(threadIdx.x, 0) = pixMin;
						deviRange.At<TPixelMap3D>(threadIdx.x, 1) = pixMax;
					}

					__syncthreads();
					//threadfence();

					if (threadIdx.x == 0)
					{
						atomicExch(pLock, 0);
					}

					surfMap3D.WritePixel2D<TPixelMap3D>(pixMap3D, iLeftX, iLeftY);
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
				m_xPars = xPars;

				// Reserve memory for min-max data
				Clu::SImageFormat xF(Kernel::Const::ThreadsPerBlockX, 2, Clu::SImageType(Clu::EPixelType::RGBA, Clu::EDataType::Single));
				m_imgRange.Create(xF);
				m_deviRange.Create(xF);

				EvalThreadConfigBlockSize(xDevice, xFormat
					, Kernel::Const::BlockSizeX, Kernel::Const::BlockSizeY
					, 0, 0, 0, 0 // Offsets
					, Kernel::Const::WarpsPerBlockX, Kernel::Const::WarpsPerBlockY
					, NumberOfRegisters
					, false // Use also partial blocks
					);
			}


			void CDriver::Process(Clu::Cuda::_CDeviceSurface& surfMap3D, const Clu::Cuda::_CDeviceSurface& surfDisparity)
			{
				if (!surfMap3D.IsOfType<TPixelMap3D>())
				{
					throw CLU_EXCEPTION("Given 3D-map image is not of the correct type");
				}

				if (!surfMap3D.IsEqualSize(surfDisparity.Format()))
				{
					throw CLU_EXCEPTION("3D-map and disparity images are not of the same size");
				}

				// Initialize min-max

				using TFloat = TPixelMap3D::TData;
				TPixelMap3D *pData = (TPixelMap3D*)m_imgRange.DataPointer();
				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					pData->r() = Clu::NumericLimits<TFloat>::Max();
					pData->g() = Clu::NumericLimits<TFloat>::Max();
					pData->b() = Clu::NumericLimits<TFloat>::Max();
					pData->a() = TFloat(0);
				}
				
				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					pData->r() = -Clu::NumericLimits<TFloat>::Max();
					pData->g() = -Clu::NumericLimits<TFloat>::Max();
					pData->b() = -Clu::NumericLimits<TFloat>::Max();
					pData->a() = TFloat(0);
				}


				m_deviRange.CopyFrom(m_imgRange);

				Clu::Cuda::MemCpyToSymbol(Kernel::c_xPars, &m_xPars, 1, 0, Clu::Cuda::ECopyType::HostToDevice);

				if (surfDisparity.IsOfType<TPixelDispEx>())
				{
					Kernel::Map<TPixelDispEx>
						CLU_KERNEL_CONFIG()
						(surfMap3D, m_deviRange, surfDisparity);
				}
				else if (surfDisparity.IsOfType<TPixelDisp>())
				{
					Kernel::Map<TPixelDisp>
						CLU_KERNEL_CONFIG()
						(surfMap3D, m_deviRange, surfDisparity);
				}
				else if (surfDisparity.IsOfType<TPixelDispF>())
				{
					Kernel::Map<TPixelDispF>
						CLU_KERNEL_CONFIG()
						(surfMap3D, m_deviRange, surfDisparity);
				}
				else
				{
					throw CLU_EXCEPTION("Given disparity image is not of the correct type");
				}



				m_deviRange.CopyInto(m_imgRange);
				m_xPars.ResetRange();

				pData = (TPixelMap3D*)m_imgRange.DataPointer();
				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					m_xPars.pixMin.r() = Clu::Min(m_xPars.pixMin.r(), pData->r());
					m_xPars.pixMin.g() = Clu::Min(m_xPars.pixMin.g(), pData->g());
					m_xPars.pixMin.b() = Clu::Min(m_xPars.pixMin.b(), pData->b());
				}

				for (int iX = 0; iX < Kernel::Const::ThreadsPerBlockX; ++iX, ++pData)
				{
					m_xPars.pixMax.r() = Clu::Max(m_xPars.pixMax.r(), pData->r());
					m_xPars.pixMax.g() = Clu::Max(m_xPars.pixMax.g(), pData->g());
					m_xPars.pixMax.b() = Clu::Max(m_xPars.pixMax.b(), pData->b());
				}

			}


		} // ImgProc
	} // Cuda
} // Clu


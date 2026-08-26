/*-------------------------------------------------------------------------
 *
 * Header CUDA functions for reconstruction
 *
 *
 * CODE by       Wei Hu
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Copyright (c) 2024, Beihang University
All rights reserved.
Contact: Weihu22@buaa.edu.cn
Codes  : XXXX
---------------------------------------------------------------------------
 */
struct GPUBufferPool;
void IRN_TV_CGLS(Geometry geo, float* projectionsU, float* projectionsV, ReconstructionPara reconP, float* result, const GpuIds& gpuids, GPUBufferPool& gpuPool);
void computeDeltas_Siddon(Geometry geo, int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source);
void computeDeltasCube(Geometry geo, int i, Point3D* xyzorigin, Point3D* source, Point3D* midPtBK, Point3D* midDirBK, Point3D* deltaU, Point3D* deltaV);
void computeBackgroundCoef(Geometry geo, int i, Point3D S, Point3D midPtBK, float* DSD, float* co);
#pragma once

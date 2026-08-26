/*-------------------------------------------------------------------------
 *
 * Header CUDA functions for backrpojection using match weights
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
#include "types_BOSLAB.h"
#include "GpuIds.hpp"

#ifndef BACKPROJECTION2_HPP
#define BACKPROJECTION2_HPP

int voxel_backprojection(float  *  projections, Geometry geo, const int diffselect, float* result, bool pseudo_mask, const GpuIds& gpuids);
void splitCTbackprojection(const GpuIds& gpuids, Geometry geo, unsigned int* split_image, unsigned int * split_projections);
void computeDeltasCube(Geometry geo, int i, Point3D* xyzorigin, Point3D* source, Point3D* midPtBK, Point3D* midDirBK, Point3D* deltaU, Point3D* deltaV);
void createGeoArray(unsigned int image_splits, Geometry geo, Geometry* geoArray);
void computeBackgroundCoef(Geometry geo, int i, Point3D S, Point3D midPtBK, float* DSD, float* co);
void computeInterp(Geometry geo, Point3D xyzOrigin, Point3D S, Point3D midPtBK, Point3D midDirBK, Point3D deltaU, Point3D deltaV);
#endif#pragma once

/*-------------------------------------------------------------------------
 *
 * CUDA functions for 3D reconstruction based on IRN_TV_CGLS algorithm
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
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "BOSLAB_common.h"
#include "reconstruction.h"
#include "gpu_buffer_pool.h"

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
				mexPrintf("Error: %s\n", msg);\
				mexPrintf("IRN_TV_CGLS: %s\n", cudaGetErrorString(__err)); \
        } \
} while (0)


__global__ void build_weights_kernel(float* W, int Nx, int Ny, int Nz, cudaTextureObject_t tex_x) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx >= Nx || idy >= Ny || idz >= Nz) return;

	int i = idz * (Nx * Ny) + idy * Nx + idx;

	float Dxx = 0.0f, Dyx = 0.0f, Dzx = 0.0f;

	// 计算 Dxx, Dyx, Dzx，使用纹理内存读取
	if (idx < Nx - 1)
		Dxx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx + 1, idy, idz);
	if (idy < Ny - 1)
		Dyx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx, idy + 1, idz);
	if (idz < Nz - 1)
		Dzx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx, idy, idz + 1);

	// Calculate weights
	W[i] = powf(Dxx * Dxx + Dyx * Dyx + Dzx * Dzx + 1e-6f, -0.25f);

}

__global__ void Axfun_kernel(float* detector,
	Geometry geo,
	const int diffselect,
	cudaTextureObject_t tex_x,
	Point3D* projParamsArrayDev) {


	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	int projNumber = blockIdx.z * blockDim.z + threadIdx.z;

	if (projNumber >= geo.numCam)
		return;

	if (u >= geo.maxnCamU || v >= geo.maxnCamV)
		return;


	size_t idx = (u * geo.maxnCamV + v) + projNumber * geo.maxnCamV *geo.maxnCamU;
	size_t idx_start = (diffselect - 1)*geo.maxnCamV*geo.maxnCamU*geo.numCam;

	Point3D uvOrigin = projParamsArrayDev[4 * projNumber];  // 6*projNumber because we have 6 Point3D values per projection
	Point3D deltaU = projParamsArrayDev[4 * projNumber + 1];
	Point3D deltaV = projParamsArrayDev[4 * projNumber + 2];
	Point3D source = projParamsArrayDev[4 * projNumber + 3];

	//printf("uvOriginx: %f\n", uvOrigin.x);

	/////// Get coordinates XYZ of pixel UV
	unsigned long pixelV = v;
	unsigned long pixelU = u;
	Point3D pixel1D;
	pixel1D.x = (uvOrigin.x + pixelU * deltaU.x + pixelV * deltaV.x);
	pixel1D.y = (uvOrigin.y + pixelU * deltaU.y + pixelV * deltaV.y);
	pixel1D.z = (uvOrigin.z + pixelU * deltaU.z + pixelV * deltaV.z);
	///////
	// Siddon's ray-voxel intersection, optimized as in doi=10.1.1.55.7516
	//////
	// Also called Jacobs algorithms
	Point3D ray;
	// vector of Xray
	ray.x = pixel1D.x - source.x;
	ray.y = pixel1D.y - source.y;
	ray.z = pixel1D.z - source.z;
	float eps = 0.001;
	ray.x = (fabsf(ray.x) < eps) ? 0 : ray.x;
	ray.y = (fabsf(ray.y) < eps) ? 0 : ray.y;
	ray.z = (fabsf(ray.z) < eps) ? 0 : ray.z;
	// This variables are ommited because
	// bx,by,bz ={0,0,0}
	// dx,dy,dz ={1,1,1}
	// compute parameter values for x-ray parametric equation. eq(3-10)
	float axm, aym, azm;
	float axM, ayM, azM;
	// In the paper Nx= number of X planes-> Nvoxel+1

	axm = fminf(__fdividef(-source.x, ray.x), __fdividef(geo.nVoxelX - source.x, ray.x));
	aym = fminf(__fdividef(-source.y, ray.y), __fdividef(geo.nVoxelY - source.y, ray.y));
	azm = fminf(__fdividef(-source.z, ray.z), __fdividef(geo.nVoxelZ - source.z, ray.z));
	axM = fmaxf(__fdividef(-source.x, ray.x), __fdividef(geo.nVoxelX - source.x, ray.x));
	ayM = fmaxf(__fdividef(-source.y, ray.y), __fdividef(geo.nVoxelY - source.y, ray.y));
	azM = fmaxf(__fdividef(-source.z, ray.z), __fdividef(geo.nVoxelZ - source.z, ray.z));

	float am = fmaxf(fmaxf(axm, aym), azm);
	float aM = fminf(fminf(axM, ayM), azM);

	// line intersects voxel space ->   am<aM
	if (am >= aM) {
		detector[idx + idx_start] = 0;
		return;
	}

	// Compute max/min image INDEX for intersection eq(11-19)
	// Discussion about ternary operator in CUDA: https://stackoverflow.com/questions/7104384/in-cuda-why-is-a-b010-more-efficient-than-an-if-else-version
	float imin, imax, jmin, jmax, kmin, kmax;
	// for X
	if (source.x < pixel1D.x) {
		imin = (am == axm) ? 1.0f : ceilf(source.x + am * ray.x);
		imax = (aM == axM) ? geo.nVoxelX : floorf(source.x + aM * ray.x);
	}
	else {
		imax = (am == axm) ? geo.nVoxelX - 1.0f : floorf(source.x + am * ray.x);
		imin = (aM == axM) ? 0.0f : ceilf(source.x + aM * ray.x);
	}
	// for Y
	if (source.y < pixel1D.y) {
		jmin = (am == aym) ? 1.0f : ceilf(source.y + am * ray.y);
		jmax = (aM == ayM) ? geo.nVoxelY : floorf(source.y + aM * ray.y);
	}
	else {
		jmax = (am == aym) ? geo.nVoxelY - 1.0f : floorf(source.y + am * ray.y);
		jmin = (aM == ayM) ? 0.0f : ceilf(source.y + aM * ray.y);
	}
	// for Z
	if (source.z < pixel1D.z) {
		kmin = (am == azm) ? 1.0f : ceilf(source.z + am * ray.z);
		kmax = (aM == azM) ? geo.nVoxelZ : floorf(source.z + aM * ray.z);
	}
	else {
		kmax = (am == azm) ? geo.nVoxelZ - 1.0f : floorf(source.z + am * ray.z);
		kmin = (aM == azM) ? 0.0f : ceilf(source.z + aM * ray.z);
	}

	// get intersection point N1. eq(20-21) [(also eq 9-10)]
	float ax, ay, az;
	ax = (source.x < pixel1D.x) ? __fdividef(imin - source.x, ray.x) : __fdividef(imax - source.x, ray.x);
	ay = (source.y < pixel1D.y) ? __fdividef(jmin - source.y, ray.y) : __fdividef(jmax - source.y, ray.y);
	az = (source.z < pixel1D.z) ? __fdividef(kmin - source.z, ray.z) : __fdividef(kmax - source.z, ray.z);

	// If its Infinite (i.e. ray is parallel to axis), make sure its positive
	ax = (isinf(ax)) ? abs(ax) : ax;
	ay = (isinf(ay)) ? abs(ay) : ay;
	az = (isinf(az)) ? abs(az) : az;


	// get index of first intersection. eq (26) and (19)
	unsigned long i, j, k;
	float aminc = fminf(fminf(ax, ay), az);
	i = (unsigned long)floorf(source.x + (aminc + am)*0.5f*ray.x);
	j = (unsigned long)floorf(source.y + (aminc + am)*0.5f*ray.y);
	k = (unsigned long)floorf(source.z + (aminc + am)*0.5f*ray.z);


	// Initialize
	float ac = am;
	//eq (28), unit anlges
	float axu, ayu, azu;
	axu = __frcp_rd(fabsf(ray.x));
	ayu = __frcp_rd(fabsf(ray.y));
	azu = __frcp_rd(fabsf(ray.z));
	// eq(29), direction of update
	float iu, ju, ku;
	iu = (source.x < pixel1D.x) ? 1.0f : -1.0f;
	ju = (source.y < pixel1D.y) ? 1.0f : -1.0f;
	ku = (source.z < pixel1D.z) ? 1.0f : -1.0f;

	float maxlength = __fsqrt_rd(ray.x*ray.x*geo.dVoxelX*geo.dVoxelX + ray.y*ray.y*geo.dVoxelY*geo.dVoxelY + ray.z*ray.z*geo.dVoxelZ*geo.dVoxelZ);
	float sum = 0.0f;
	unsigned long Np = (imax - imin + 1) + (jmax - jmin + 1) + (kmax - kmin + 1); // Number of intersections
	// Go iterating over the line, intersection by intersection. If double point, no worries, 0 will be computed


	switch (diffselect) {
	case 0:
		for (unsigned long ii = 0; ii < Np; ii++) {
			if (ax == aminc) {
				sum += (ax - ac)*tex3D<float>(tex_x, i, j, k);
				i = i + iu;
				ac = ax;
				ax += axu;
			}
			else if (ay == aminc) {
				sum += (ay - ac)*tex3D<float>(tex_x, i, j, k);
				j = j + ju;
				ac = ay;
				ay += ayu;
			}
			else if (az == aminc) {
				sum += (az - ac)*tex3D<float>(tex_x, i, j, k);
				k = k + ku;
				ac = az;
				az += azu;
			}
			aminc = fminf(fminf(ax, ay), az);
		}
		detector[idx + idx_start] = sum * maxlength;
		break;
	case 1:
		for (unsigned long ii = 0; ii < Np; ii++) {
			if (ax == aminc) {
				sum += (ax - ac)*(tex3D<float>(tex_x, i + 1, j, k) - tex3D<float>(tex_x, i - 1, j, k)) / 2 / geo.dVoxelX;
				i = i + iu;
				ac = ax;
				ax += axu;
			}
			else if (ay == aminc) {
				sum += (ay - ac)*(tex3D<float>(tex_x, i + 1, j, k) - tex3D<float>(tex_x, i - 1, j, k)) / 2 / geo.dVoxelX;
				j = j + ju;
				ac = ay;
				ay += ayu;
			}
			else if (az == aminc) {
				sum += (az - ac)*(tex3D<float>(tex_x, i + 1, j, k) - tex3D<float>(tex_x, i - 1, j, k)) / 2 / geo.dVoxelX;
				k = k + ku;
				ac = az;
				az += azu;
			}
			aminc = fminf(fminf(ax, ay), az);
		}
		detector[idx + idx_start] = sum * maxlength;
		break;
	case 2:
		for (unsigned long ii = 0; ii < Np; ii++) {
			if (ax == aminc) {
				sum += (ax - ac)*(tex3D<float>(tex_x, i, j + 1, k) - tex3D<float>(tex_x, i, j - 1, k)) / 2 / geo.dVoxelY;
				i = i + iu;
				ac = ax;
				ax += axu;
			}
			else if (ay == aminc) {
				sum += (ay - ac)*(tex3D<float>(tex_x, i, j + 1, k) - tex3D<float>(tex_x, i, j - 1, k)) / 2 / geo.dVoxelY;
				j = j + ju;
				ac = ay;
				ay += ayu;
			}
			else if (az == aminc) {
				sum += (az - ac)*(tex3D<float>(tex_x, i, j + 1, k) - tex3D<float>(tex_x, i, j - 1, k)) / 2 / geo.dVoxelY;
				k = k + ku;
				ac = az;
				az += azu;
			}
			aminc = fminf(fminf(ax, ay), az);
		}
		detector[idx + idx_start] = sum * maxlength;
		break;
	case 3:
		for (unsigned long ii = 0; ii < Np; ii++) {
			if (ax == aminc) {
				sum += (ax - ac)*(tex3D<float>(tex_x, i, j, k + 1) - tex3D<float>(tex_x, i, j, k - 1)) / 2 / geo.dVoxelZ;
				i = i + iu;
				ac = ax;
				ax += axu;
			}
			else if (ay == aminc) {
				sum += (ay - ac)*(tex3D<float>(tex_x, i, j, k + 1) - tex3D<float>(tex_x, i, j, k - 1)) / 2 / geo.dVoxelZ;
				j = j + ju;
				ac = ay;
				ay += ayu;
			}
			else if (az == aminc) {
				sum += (az - ac)*(tex3D<float>(tex_x, i, j, k + 1) - tex3D<float>(tex_x, i, j, k - 1)) / 2 / geo.dVoxelZ;
				k = k + ku;
				ac = az;
				az += azu;
			}
			aminc = fminf(fminf(ax, ay), az);
		}
		detector[idx + idx_start] = sum * maxlength;
		break;
	default:
		break;
	}
}



__global__ void Lx_kernel(float* out, float* W, cudaTextureObject_t tex_x, int Nx, int Ny, int Nz, float lambda_sqrt) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx >= Nx || idy >= Ny || idz >= Nz) return;

	int i = idz * (Nx * Ny) + idy * Nx + idx;

	float Dxx = 0.0f, Dyx = 0.0f, Dzx = 0.0f;

	// 计算 Dxx, Dyx, Dzx，使用纹理内存读取
	if (idx < Nx - 1)
		Dxx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx + 1, idy, idz);
	if (idy < Ny - 1)
		Dyx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx, idy + 1, idz);
	if (idz < Nz - 1)
		Dzx = tex3D<float>(tex_x, idx, idy, idz) - tex3D<float>(tex_x, idx, idy, idz + 1);

	// 应用权重
	out[i] = W[i] * Dxx * lambda_sqrt;
	out[i + Nx * Ny * Nz] = W[i] * Dyx * lambda_sqrt;
	out[i + 2 * Nx * Ny * Nz] = W[i] * Dzx * lambda_sqrt;
}


__global__ void sub_kernel(float* out, float* A, float* B, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {
		out[idx] = A[idx] - B[idx];
	}
}

__global__ void negate4D_kernel(float* r_aux_2, const float* prox_aux_2, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements) {
		r_aux_2[idx] = -prox_aux_2[idx];
	}
}


__global__ void Atbfun_kernel(float* image, Geometry geo, const int currProjSetNumber,
	cudaTextureObject_t tex_r_aux_1, Point3D* projParamsArray2Dev, float* projCoeffArray2Dev, const int diffselect)
{
	unsigned long long indY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long long indX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long indZ = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned long long startInddiffselect = (diffselect - 1) * geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
	unsigned long long startInddiffselectProj = (diffselect - 1) * geo.numCam;
	//Make sure we don't go out of bounds
	if (indX >= geo.nVoxelX || indY >= geo.nVoxelY || indZ >= geo.nVoxelZ)
		return;

	// We'll keep a local auxiliary array of values of a column of voxels that this thread will update
	//float voxelColumn[VOXELS_PER_THREAD];
	float voxelColumn;
	unsigned long long idx = indZ * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + indY * (unsigned long long)geo.nVoxelX + indX;
	voxelColumn = image[idx + startInddiffselect];

	unsigned long indAlpha = currProjSetNumber;

	Point3D xyzOrigin = projParamsArray2Dev[6 * currProjSetNumber];  // 6*projNumber because we have 6 Point3D values per projection
	Point3D S = projParamsArray2Dev[6 * currProjSetNumber + 1];
	Point3D midPtBK = projParamsArray2Dev[6 * currProjSetNumber + 2];
	Point3D midDirBK = projParamsArray2Dev[6 * currProjSetNumber + 3];
	Point3D deltaU = projParamsArray2Dev[6 * currProjSetNumber + 4];
	Point3D deltaV = projParamsArray2Dev[6 * currProjSetNumber + 5];


	float DSD = projCoeffArray2Dev[2 * currProjSetNumber];
	float co = projCoeffArray2Dev[2 * currProjSetNumber + 1];

	Point3D P;
	P.x = xyzOrigin.x + indX;
	P.y = xyzOrigin.y + indY;
	P.z = xyzOrigin.z + indZ;

	// This is the vector defining the line from the source to the Voxel
	float vectX, vectY, vectZ;
	vectX = (P.x - S.x);
	vectY = (P.y - S.y);
	vectZ = (P.z - S.z);

	// Get the coordinates in the detector UV where the mid point of the voxel is projected.
	float t, fm, fz;
	fm = midDirBK.x*vectX + midDirBK.y*vectY + midDirBK.z*vectZ;
	if (fm == 0) {
		return;
	}
	else {
		fz = (midPtBK.x - P.x)*midDirBK.x + (midPtBK.y - P.y)*midDirBK.y + (midPtBK.z - P.z)*midDirBK.z;
		t = __fdividef(fz, fm);
	}

	Point3D pointBK;
	pointBK.x = vectX * t + P.x;
	pointBK.y = vectY * t + P.y;
	pointBK.z = vectZ * t + P.z;

	float u, v;
	if (deltaU.x != 0) {
		u = (pointBK.x - midPtBK.x) / deltaU.x + geo.maxnCamU * 0.5f;
	}
	else if (deltaU.z != 0) {
		u = (pointBK.z - midPtBK.z) / deltaU.z + geo.maxnCamU * 0.5f;
	}
	else if (deltaU.y != 0) {
		u = (pointBK.y - midPtBK.y) / deltaU.y + geo.maxnCamU * 0.5f;
	}

	if (deltaV.y != 0) {
		v = (pointBK.y - midPtBK.y) / deltaV.y + geo.maxnCamV * 0.5f;
	}
	else if (deltaU.x != 0) {
		v = (pointBK.x - midPtBK.x) / deltaV.x + geo.maxnCamV * 0.5f;
	}
	else if (deltaU.z != 0) {
		v = (pointBK.z - midPtBK.z) / deltaV.z + geo.maxnCamV * 0.5f;
	}

	if (u >= geo.maxnCamU || v >= geo.maxnCamV) return;//break;

	float sample = tex3D<float>(tex_r_aux_1, v, u, indAlpha + 0.5f + startInddiffselectProj);
	float weight = 0;
	float L, lsq;

	L = __fsqrt_rd((S.x - pointBK.x)*(S.x - pointBK.x) + (S.y - pointBK.y)*(S.y - pointBK.y) + (S.z - pointBK.z)*(S.z - pointBK.z)); // Sz=0 always.
	lsq = (S.x - P.x)*(S.x - P.x)
		+ (S.y - P.y)*(S.y - P.y)
		+ (S.z - P.z)*(S.z - P.z);
	weight = __fdividef(L*L*L, (DSD*lsq));

	voxelColumn += sample * weight * co;

	image[idx + startInddiffselect] = voxelColumn;

}  // END kernelPixelBackprojectionFDK


__global__ void matrixDiffTMultiply(float* imagediff, Geometry geo, float* image, const int diffselect) {
	unsigned long long indY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long long indX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long indZ = blockIdx.z * blockDim.z + threadIdx.z;  // This is only STARTING z index of the column of voxels that the thread will handle
	unsigned long long startInddiffselect = (diffselect - 1) * geo.nVoxelX * geo.nVoxelY * geo.nVoxelZ;
	//Make sure we don't go out of bounds
	if (indX >= geo.nVoxelX || indY >= geo.nVoxelY || indZ >= geo.nVoxelZ)
		return;
	unsigned long long idx0;
	idx0 = indZ * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + indY * (unsigned long long)geo.nVoxelX + indX;

	unsigned long long idx1, idx2, i, j, k;
	float co0, co1, co2;

	switch (diffselect) {
	case 0:
		imagediff[idx0] = image[idx0];
		break;
	case 1:
		if (indX == 0) {
			co0 = __fdividef(-0.5, geo.dVoxelX);

			i = indX + 1;
			j = indY;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(-0.5, geo.dVoxelX);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else if (indX == geo.nVoxelX - 1) {
			co0 = __fdividef(0.5, geo.dVoxelX);

			i = indX - 1;
			j = indY;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelX);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else {
			i = indX - 1;
			j = indY;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelX);

			i = indX + 1;
			j = indY;
			k = indZ;
			idx2 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co2 = __fdividef(-0.5, geo.dVoxelX);
			imagediff[idx0] = imagediff[idx0] + image[idx1 + startInddiffselect] * co1 + image[idx2 + startInddiffselect] * co2;
		}
		break;
	case 2:
		if (indY == 0) {
			co0 = __fdividef(-0.5, geo.dVoxelY);

			i = indX;
			j = indY + 1;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(-0.5, geo.dVoxelY);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else if (indY == geo.nVoxelY - 1) {
			co0 = __fdividef(0.5, geo.dVoxelY);

			i = indX;
			j = indY - 1;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelY);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else {
			i = indX;
			j = indY - 1;
			k = indZ;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelY);

			i = indX;
			j = indY + 1;
			k = indZ;
			idx2 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co2 = __fdividef(-0.5, geo.dVoxelY);
			imagediff[idx0] = imagediff[idx0] + image[idx1 + startInddiffselect] * co1 + image[idx2 + startInddiffselect] * co2;
		}
		break;
	case 3:

		if (indZ == 0) {

			co0 = __fdividef(-0.5, geo.dVoxelZ);

			i = indX;
			j = indY;
			k = indZ + 1;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(-0.5, geo.dVoxelZ);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else if (indZ == geo.nVoxelZ - 1) {

			co0 = __fdividef(0.5, geo.dVoxelZ);

			i = indX;
			j = indY;
			k = indZ - 1;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelZ);
			imagediff[idx0] = imagediff[idx0] + image[idx0 + startInddiffselect] * co0 + image[idx1 + startInddiffselect] * co1;
		}
		else {
			i = indX;
			j = indY;
			k = indZ - 1;
			idx1 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co1 = __fdividef(0.5, geo.dVoxelZ);

			i = indX;
			j = indY;
			k = indZ + 1;
			idx2 = k * (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY + j * (unsigned long long)geo.nVoxelX + i;
			co2 = __fdividef(-0.5, geo.dVoxelZ);
			imagediff[idx0] = imagediff[idx0] + image[idx1 + startInddiffselect] * co1 + image[idx2 + startInddiffselect] * co2;
		}
		break;
	default:
		break;
	}
}



__global__ void Ltx_kernel(float *out, float *W, float *x, int dim1, int dim2, int dim3, float lambda_sqrt) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	float dxt = 0.0f;
	float dyt = 0.0f;
	float dzt = 0.0f;
	// Make sure the thread is within bounds
	if (idx >= 1 && idx < dim1 - 1) {
		dxt = W[(idx)* dim2 * dim3 + idy * dim3 + idz] * x[(idx)* dim2 * dim3 + idy * dim3 + idz + 2 * dim1 * dim2*dim3] - W[(idx - 1) * dim2 * dim3 + idy * dim3 + idz] * x[(idx - 1)* dim2 * dim3 + idy * dim3 + idz + 2 * dim1 * dim2*dim3];
	}
	else if (idx == dim1 - 1) {
		dxt = -W[(dim1 - 2) * dim2 * dim3 + idy * dim3 + idz] * x[(dim1 - 2) * dim2 * dim3 + idy * dim3 + idz + 2 * dim1 * dim2*dim3];
	}
	else if (idx == 0) {
		dxt = W[(idx)* dim2 * dim3 + idy * dim3 + idz] * x[(idx)* dim2 * dim3 + idy * dim3 + idz + 2 * dim1 * dim2*dim3];
	}

	if (idy >= 1 && idy < dim2 - 1) {
		dyt = W[idx * dim2 * dim3 + (idy)* dim3 + idz] * x[idx * dim2 * dim3 + (idy)* dim3 + idz + dim1 * dim2*dim3] - W[idx * dim2 * dim3 + (idy - 1) * dim3 + idz] * x[idx * dim2 * dim3 + (idy - 1) * dim3 + idz + dim1 * dim2*dim3];
	}
	else if (idy == dim2 - 1) {
		dyt = -W[idx * dim2 * dim3 + (dim2 - 2) * dim3 + idz] * x[idx * dim2 * dim3 + (dim2 - 2) * dim3 + idz + dim1 * dim2*dim3];
	}
	else if (idy == 0) {
		dyt = W[idx * dim2 * dim3 + (idy)* dim3 + idz] * x[idx * dim2 * dim3 + (idy)* dim3 + idz + dim1 * dim2*dim3];
	}

	if (idz >= 1 && idz < dim3 - 1) {
		dzt = W[idx * dim2 * dim3 + idy * dim3 + (idz)] * x[idx * dim2 * dim3 + idy * dim3 + (idz)] - W[idx * dim2 * dim3 + idy * dim3 + (idz - 1)] * x[idx * dim2 * dim3 + idy * dim3 + (idz - 1)];
	}
	else if (idz == dim3 - 1) {
		dzt = -W[idx * dim2 * dim3 + idy * dim3 + (dim3 - 2)] * x[idx * dim2 * dim3 + idy * dim3 + (dim3 - 2)];
	}
	else if (idz == 0) {
		dzt = W[idx * dim2 * dim3 + idy * dim3 + (idz)] * x[idx * dim2 * dim3 + idy * dim3 + (idz)];
	}


	// Store the result in the output array
	out[idx * dim2 * dim3 + idy * dim3 + idz] = (dxt + dyt + dzt)*lambda_sqrt;
}

__global__ void add_kernel(float* out, float* A, float* B, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		out[idx] = A[idx] + B[idx];
	}
}

__global__ void norm2square_kernel(float* out, float* p, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numElements) { return; }
	float value = p[idx];

	// 原子加法，避免竞态条件
	atomicAdd(out, value * value);
}


__global__ void calcAlpha_kernel(float* d_alpha, float* d_gamma, float* d_gamma2, float* d_gamma3) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 1) {
		if ((d_gamma2[idx] + d_gamma3[idx]) != 0.0f) {
			d_alpha[idx] = d_gamma[idx] / (d_gamma2[idx] + d_gamma3[idx]);
		}
		else {
			d_alpha[idx] = 0.0f;  // 处理除0的情况
		}
	}
}

__global__ void updateX_kernel(float* d_x, const float* d_p, float* alpha, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		d_x[idx] = d_x[idx] + alpha[0] * d_p[idx];  // 执行计算
	}
}

__global__ void updateP_kernel(float* d_p, const float* d_s, float* beta, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		d_p[idx] = d_s[idx] + beta[0] * d_p[idx];  // 执行计算
	}
}

__global__ void updateXsub_kernel(float* d_x, const float* d_p, float* alpha, int numElements) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElements) {
		d_x[idx] = d_x[idx] - alpha[0] * d_p[idx];  // 执行计算
	}
}

__global__ void sqrt_kernel(float* out, float* p) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		out[0] = sqrt(p[0]);
	}
}

__global__ void update_resL2_kernel(float *d_resL2, float *d_gamma, int iter) {
	if (threadIdx.x == 0 && blockIdx.x == 0) { // 保证只有一个线程执行此操作
		d_resL2[iter - 1] = d_gamma[0];  // 更新 d_resL2[iter]，将 d_gamma[0] 的值复制过去
	}
}

__global__ void check_condition_kernel(bool *d_condition, float *d_resL2, int niter_break, int iter) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		if (iter == 1) {
			if (niter_break != 0 && ((iter) % niter_break) != 1) {
				d_condition[0] = true;  // Condition met, mark as true
			}
			else {
				d_condition[0] = false;  // Condition not met, mark as false
			}
		}
		else {
			if (niter_break != 0 && ((iter) % niter_break) != 1 && (d_resL2[iter - 1] > d_resL2[iter - 2] + 1e-4f)) {
				d_condition[0] = true;  // Condition met, mark as true
			}
			else {
				d_condition[0] = false;  // Condition not met, mark as false
			}
		}
	}
}

__global__ void div_kernel(float* out, float* A, float* B) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		if (B[0] == 0) {
			out[0] = 0;
		}
		else {
			out[0] = A[0] / B[0];
		}
	}
}

__global__ void equal_kernel(float* out, float* A) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		out[0] = A[0];
	}
}

void IRN_TV_CGLS(Geometry geo, float* projectionsU, float* projectionsV, ReconstructionPara reconP, float* result, const GpuIds& gpuids, GPUBufferPool& gpuPool) {
	auto t0 = std::chrono::high_resolution_clock::now();
	// Prepare for MultiGPU

	cudaSetDevice(0);
	// for parallel calculating
	

	cudaMemcpy3DParms copyParams = { 0 };
	cudaMemcpy3DParms copyParams_p = { 0 };
	cudaMemset(gpuPool.d_x, 0, gpuPool.voxelSize * sizeof(float));
	cudaMemset(gpuPool.d_resL2, 0, reconP.niter * sizeof(float));

	dim3 blockSize(4, 4, 4);  // Block size for 3D grid
	dim3 gridSize_flow((geo.nVoxelX + blockSize.x - 1) / blockSize.x,
		(geo.nVoxelY + blockSize.y - 1) / blockSize.y,
		(geo.nVoxelZ + blockSize.z - 1) / blockSize.z);
	dim3 gridSize_proj((geo.maxnCamU + blockSize.x - 1) / blockSize.x,
		(geo.maxnCamV + blockSize.y - 1) / blockSize.y,
		(geo.numCam + blockSize.z - 1) / blockSize.z);
	dim3 gridSize_3xproj((geo.maxnCamU + blockSize.x - 1) / blockSize.x,
		(geo.maxnCamV + blockSize.y - 1) / blockSize.y,
		(3 * geo.numCam + blockSize.z - 1) / blockSize.z);
	int blockSize_1D = 256;
	int gridSize_4Dflow = (3 * geo.nVoxelX*geo.nVoxelY*geo.nVoxelZ + blockSize_1D - 1) / blockSize_1D;
	int gridSize_4Dproj = (3 * geo.maxnCamU*geo.maxnCamV* geo.numCam + blockSize_1D - 1) / blockSize_1D;
	int gridSize_3Dflow = (geo.nVoxelX*geo.nVoxelY*geo.nVoxelZ + blockSize_1D - 1) / blockSize_1D;
	int gridSize_3Dproj = (geo.maxnCamU*geo.maxnCamV * geo.numCam + blockSize_1D - 1) / blockSize_1D;

	int iter = 0;
	int remember = -1;
	bool early_stop = false;

	while (iter < reconP.niter) {
		copyParams.srcPtr = make_cudaPitchedPtr(gpuPool.d_x, geo.nVoxelX * sizeof(float), geo.nVoxelX, geo.nVoxelY);
		copyParams.dstArray = gpuPool.d_x_array;
		copyParams.extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
		copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&copyParams);
		cudaCheckErrors("tex_x cudaMemcpy3D failed");

		// Call the CUDA kernel to build weights
		build_weights_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_W, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, gpuPool.tex_x);
		cudaDeviceSynchronize();

		cudaCheckErrors("build_weights_kernel fail");

		// Call Axfun kernel
		//cudaMemset(gpuPool.d_prox_aux_1, 0, 3 * gpuPool.projSize * sizeof(float));

		Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		cudaDeviceSynchronize();
		cudaCheckErrors("Axfun_kernel fail");

		// Call Lx kernel
		float lambda_sqrt = sqrt(reconP.lambda);
		Lx_kernel << <gridSize_flow, blockSize>> > (gpuPool.d_prox_aux_2, gpuPool.d_W, gpuPool.tex_x, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
		cudaDeviceSynchronize();
		cudaCheckErrors("Lx_kernel fail");

		// Call sub kernel
		sub_kernel << <gridSize_4Dproj, blockSize_1D>> > (gpuPool.d_r_aux_1, gpuPool.d_projectionsob, gpuPool.d_prox_aux_1, 3 * gpuPool.projSize);
		cudaDeviceSynchronize();
		cudaCheckErrors("sub_kernel failed");

		// Call negate4D kernel
		negate4D_kernel << <gridSize_4Dflow, blockSize_1D>> > (gpuPool.d_r_aux_2, gpuPool.d_prox_aux_2, 3 * gpuPool.voxelSize);
		cudaDeviceSynchronize();
		cudaCheckErrors("negate4D_kernel failed");

		// Call Atbfun kernel
		copyParams_p.srcPtr = make_cudaPitchedPtr(gpuPool.d_r_aux_1, geo.maxnCamV * sizeof(float), geo.maxnCamV, geo.maxnCamU);
		copyParams_p.dstArray = gpuPool.d_p_array;
		copyParams_p.extent = make_cudaExtent(geo.maxnCamV, geo.maxnCamU, 3 * geo.numCam);
		copyParams_p.kind = cudaMemcpyDeviceToDevice;  // 设备到设备内存复制
		cudaMemcpy3D(&copyParams_p);
		cudaCheckErrors("tex_p cudaMemcpy3D failed");

		cudaMemset(gpuPool.d_image, 0, 3 * gpuPool.voxelSize * sizeof(float));
		cudaMemset(gpuPool.d_p_aux_1, 0, gpuPool.voxelSize * sizeof(float));

		for (unsigned int proj_j = 0; proj_j < geo.numCam; proj_j++) {
			Atbfun_kernel << <gridSize_flow, blockSize>> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 1);
			Atbfun_kernel << <gridSize_flow, blockSize>> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 2);
			Atbfun_kernel << <gridSize_flow, blockSize>> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 3);
		}
		cudaDeviceSynchronize();

		matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 1);
		matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 2);
		matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 3);
		cudaDeviceSynchronize();
		cudaCheckErrors("Atbfun_kernel failed");

		// Call Ltx kernel
		Ltx_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_p_aux_2, gpuPool.d_W, gpuPool.d_r_aux_2, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
		cudaDeviceSynchronize();
		cudaCheckErrors("Ltx_kernel failed");

		// Call add kernel
		add_kernel << <gridSize_3Dflow, blockSize_1D >> > (gpuPool.d_p, gpuPool.d_p_aux_1, gpuPool.d_p_aux_2, gpuPool.voxelSize);
		cudaDeviceSynchronize();
		cudaCheckErrors("add_kernel failed");

		// Call norm2 kernel
		cudaMemset(gpuPool.d_gamma, 0, sizeof(float));
		norm2square_kernel << <gridSize_3Dflow, blockSize_1D >> > (gpuPool.d_gamma, gpuPool.d_p, gpuPool.voxelSize);
		cudaDeviceSynchronize();
		cudaCheckErrors("norm2square_kernel failed");

		// while
		while (iter < reconP.niter) {
			iter++;
			// Call Axfun kernel		
			copyParams.srcPtr = make_cudaPitchedPtr(gpuPool.d_p, geo.nVoxelX * sizeof(float), geo.nVoxelX, geo.nVoxelY);
			copyParams.dstArray = gpuPool.d_x_array;
			copyParams.extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3D(&copyParams);
			cudaCheckErrors("tex_x cudaMemcpy3D failed");

			//cudaMemset(gpuPool.d_q_aux_1, 0, 3 * gpuPool.projSize * sizeof(float));

			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			cudaDeviceSynchronize();

			cudaCheckErrors("while Axfun_kernel fail");

			// Call Lx kernel
			Lx_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_q_aux_2, gpuPool.d_W, gpuPool.tex_x, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
			cudaDeviceSynchronize();
			cudaCheckErrors("while Lx_kernel fail");

			// Call alpha kernel
			cudaMemset(gpuPool.d_gamma_q_aux_1, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dproj, blockSize_1D>> > (gpuPool.d_gamma_q_aux_1, gpuPool.d_q_aux_1, 3 * gpuPool.projSize);
			cudaMemset(gpuPool.d_gamma_q_aux_2, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dflow, blockSize_1D>> > (gpuPool.d_gamma_q_aux_2, gpuPool.d_q_aux_2, 3 * gpuPool.voxelSize);
			cudaDeviceSynchronize();
			calcAlpha_kernel << <1, 1 >> > (gpuPool.d_alpha, gpuPool.d_gamma, gpuPool.d_gamma_q_aux_1, gpuPool.d_gamma_q_aux_2);
			cudaDeviceSynchronize();
			cudaCheckErrors("while alpha_kernel failed");

			// Call updateX kernel
			updateX_kernel << <gridSize_3Dflow, blockSize_1D >> > (gpuPool.d_x, gpuPool.d_p, gpuPool.d_alpha, gpuPool.voxelSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while updateX_kernel failed");

			// Call calcAux=sub + Ax kernel
			copyParams.srcPtr = make_cudaPitchedPtr(gpuPool.d_x, geo.nVoxelX * sizeof(float), geo.nVoxelX, geo.nVoxelY);
			copyParams.dstArray = gpuPool.d_x_array;
			copyParams.extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3D(&copyParams);
			cudaCheckErrors("tex_x cudaMemcpy3D failed");

			//cudaMemset(gpuPool.d_aux_m, 0, 3 * gpuPool.projSize * sizeof(float));

			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calcuAux-Axfun_kernel fail");

			sub_kernel << <gridSize_4Dproj, blockSize_1D >> > (gpuPool.d_aux, gpuPool.d_projectionsob, gpuPool.d_aux_m, 3 * gpuPool.projSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calcuAux-sub_kernel failed");

			//  Calc resL2
			cudaMemset(gpuPool.d_aux_gamma, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dproj, blockSize_1D >> > (gpuPool.d_aux_gamma, gpuPool.d_aux, 3 * gpuPool.projSize);
			cudaDeviceSynchronize();
			sqrt_kernel << <1, 1 >> > (gpuPool.d_aux_gamma, gpuPool.d_aux_gamma);
			cudaDeviceSynchronize();
			update_resL2_kernel << <1, 1 >> > (gpuPool.d_resL2, gpuPool.d_aux_gamma, iter);
			cudaDeviceSynchronize();
			cudaCheckErrors("while  resL2_kernel failed");


			// Call check_condition kernel
			check_condition_kernel << <1, 1 >> > (gpuPool.d_condition, gpuPool.d_resL2, reconP.niter_break, iter);
			cudaDeviceSynchronize();
			cudaCheckErrors("while  check_condition_kernel failed");
			cudaMemcpy(gpuPool.h_condition, gpuPool.d_condition, sizeof(bool), cudaMemcpyDeviceToHost);
			//// if condition is true
			if (*gpuPool.h_condition)
			{
				// Call updateXsub kernel
				updateXsub_kernel << <gridSize_3Dflow, blockSize_1D >> > (gpuPool.d_x, gpuPool.d_p, gpuPool.d_alpha, gpuPool.voxelSize);
				cudaDeviceSynchronize();
				cudaCheckErrors("while updateXsub_kernel failed");
				// algorithm stoped declare
				if (remember == iter) {
					cudaMemcpy(gpuPool.h_x, gpuPool.d_x, gpuPool.voxelSize * sizeof(float), cudaMemcpyDeviceToHost);
					early_stop = true;
					break;
				}
				remember = iter;
				iter--;
				mexPrintf("Orthogonality lost, restarting at iteration %d \n", iter);
				break;
			}
			// if If step is adecuate, then continue withg CGLS
			// r_aux_1
			updateXsub_kernel << <gridSize_4Dproj, blockSize_1D >> > (gpuPool.d_r_aux_1, gpuPool.d_q_aux_1, gpuPool.d_alpha, 3 * gpuPool.projSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while update r_aux_1 --> updateXsub_kernel failed");
			// r_aux_2
			updateXsub_kernel << <gridSize_4Dflow, blockSize_1D >> > (gpuPool.d_r_aux_2, gpuPool.d_q_aux_2, gpuPool.d_alpha, 3 * gpuPool.voxelSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while update r_aux_2 --> updateXsub_kernel failed");

			// s_aux_1
			copyParams_p.srcPtr = make_cudaPitchedPtr(gpuPool.d_r_aux_1, geo.maxnCamV * sizeof(float), geo.maxnCamV, geo.maxnCamU);
			copyParams_p.dstArray = gpuPool.d_p_array;
			copyParams_p.extent = make_cudaExtent(geo.maxnCamV, geo.maxnCamU, 3 * geo.numCam);
			copyParams_p.kind = cudaMemcpyDeviceToDevice;  // 设备到设备内存复制
			cudaMemcpy3D(&copyParams_p);
			cudaCheckErrors("tex_p cudaMemcpy3D failed");

			cudaMemset(gpuPool.d_image, 0, 3 * geo.nVoxelX* geo.nVoxelY * geo.nVoxelZ * sizeof(float));
			cudaMemset(gpuPool.d_s_aux_1, 0, geo.nVoxelX* geo.nVoxelY * geo.nVoxelZ * sizeof(float));

			for (unsigned int proj_j = 0; proj_j < geo.numCam; proj_j++) {
				Atbfun_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 1);
				Atbfun_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 2);
				Atbfun_kernel << <gridSize_flow, blockSize >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 3);
				cudaDeviceSynchronize();
			}

			matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 1);
			matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 2);
			matrixDiffTMultiply << <gridSize_flow, blockSize >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 3);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calc s_aux_1-- > Atbfun_kernel failed");


			// s_aux_2
			//cudaMemset(gpuPool.d_s_aux_2, 0, sizeof(float));
			Ltx_kernel << <gridSize_flow, blockSize>> > (gpuPool.d_s_aux_2, gpuPool.d_W, gpuPool.d_r_aux_2, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calc s_aux_2-- >Ltx_kernel failed");


			// s
			//cudaMemset(gpuPool.d_s, 0, sizeof(float));
			add_kernel << <gridSize_3Dflow, blockSize_1D>> > (gpuPool.d_s, gpuPool.d_s_aux_1, gpuPool.d_s_aux_2, gpuPool.voxelSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calc s-- >add_kernel failed");


			// gamma1
			cudaMemset(gpuPool.d_gamma1, 0, sizeof(float));
			norm2square_kernel << <gridSize_3Dflow, blockSize_1D >> > (gpuPool.d_gamma1, gpuPool.d_s, gpuPool.voxelSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calc gamma1-- >norm2square_kernel failed");


			// beta
			div_kernel << <1, 1 >> > (gpuPool.d_beta, gpuPool.d_gamma1, gpuPool.d_gamma);
			cudaDeviceSynchronize();
			cudaCheckErrors("while calc beta-- >div_kernel failed");


			// update gamma
			equal_kernel << <1, 1>> > (gpuPool.d_gamma, gpuPool.d_gamma1);
			cudaDeviceSynchronize();
			cudaCheckErrors("while update gamma-- >equal_kernel failed");


			// update p
			updateP_kernel << <gridSize_3Dflow, blockSize_1D>> > (gpuPool.d_p, gpuPool.d_s, gpuPool.d_beta, gpuPool.voxelSize);
			cudaDeviceSynchronize();
			cudaCheckErrors("while update p-- >updateP_kernel failed");

			//
			if (reconP.niter_break != 0 && iter % reconP.niter_break == 0) {
				break;
			}
		}
		if (early_stop) {
			mexPrintf("[GPU] early_stop \n");
			break;
		}
	}

	// =====================================================
	// ================== output ===================
	// =====================================================


	cudaMemcpy(
		result,//gpuPool.h_x,
		gpuPool.d_x,
		gpuPool.voxelSize * sizeof(float),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	auto t1 = std::chrono::high_resolution_clock::now();
	double cgls_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	mexPrintf("total CGLS time = %.3f ms\n", cgls_ms);

}
void computeDeltas_Siddon(Geometry geo, int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source) {
	// The obkjective is to get a position of the detector in a coordinate system where:
	// 1-units are voxel size (in each direction can be different)
	// 2-The image has the its first voxel at (0,0,0)

	//optical center point
	Point3D S;
	S.x = geo.OcrX[i];
	S.y = geo.OcrY[i];
	S.z = geo.OcrZ[i];

	//background point
	Point3D Pfinal, Pfinalu0, Pfinalv0;

	Pfinal.x = geo.PbkCornerX[i * 4];
	Pfinal.y = geo.PbkCornerY[i * 4];
	Pfinal.z = geo.PbkCornerZ[i * 4];

	Pfinalv0.x = (geo.PbkCornerX[i * 4 + 1] - geo.PbkCornerX[i * 4]) / (geo.nCamV[i] - 1);
	Pfinalv0.y = (geo.PbkCornerY[i * 4 + 1] - geo.PbkCornerY[i * 4]) / (geo.nCamV[i] - 1);
	Pfinalv0.z = (geo.PbkCornerZ[i * 4 + 1] - geo.PbkCornerZ[i * 4]) / (geo.nCamV[i] - 1);

	Pfinalu0.x = (geo.PbkCornerX[i * 4 + 2] - geo.PbkCornerX[i * 4]) / (geo.nCamU[i] - 1);
	Pfinalu0.y = (geo.PbkCornerY[i * 4 + 2] - geo.PbkCornerY[i * 4]) / (geo.nCamU[i] - 1);
	Pfinalu0.z = (geo.PbkCornerZ[i * 4 + 2] - geo.PbkCornerZ[i * 4]) / (geo.nCamU[i] - 1);

	// As we want the (0,0,0) to be in a corner of the image, we need to translate everything (after rotation);
	Pfinal.x = Pfinal.x + geo.sVoxelX / 2;      Pfinal.y = Pfinal.y + geo.sVoxelY / 2;          Pfinal.z = Pfinal.z + geo.sVoxelZ / 2;
	S.x = S.x + geo.sVoxelX / 2;          S.y = S.y + geo.sVoxelY / 2;              S.z = S.z + geo.sVoxelZ / 2;

	//4. Scale everything so dVoxel==1
	Pfinal.x = Pfinal.x / geo.dVoxelX;      Pfinal.y = Pfinal.y / geo.dVoxelY;        Pfinal.z = Pfinal.z / geo.dVoxelZ;
	Pfinalu0.x = Pfinalu0.x / geo.dVoxelX;    Pfinalu0.y = Pfinalu0.y / geo.dVoxelY;      Pfinalu0.z = Pfinalu0.z / geo.dVoxelZ;
	Pfinalv0.x = Pfinalv0.x / geo.dVoxelX;    Pfinalv0.y = Pfinalv0.y / geo.dVoxelY;      Pfinalv0.z = Pfinalv0.z / geo.dVoxelZ;
	S.x = S.x / geo.dVoxelX;          S.y = S.y / geo.dVoxelY;            S.z = S.z / geo.dVoxelZ;

	// return
	*uvorigin = Pfinal;

	*deltaU = Pfinalu0;
	*deltaV = Pfinalv0;

	*source = S;
}

void computeDeltasCube(Geometry geo, int i, Point3D* xyzorigin, Point3D* source, Point3D* midPtBK, Point3D* midDirBK, Point3D* deltaU, Point3D* deltaV)
{

	// initialize points with double precision
	Point3D P, Px, Py, Pz;

	// Get coords of Img(0,0,0)
	P.x = -(geo.sVoxelX / 2 - geo.dVoxelX / 2);
	P.y = -(geo.sVoxelY / 2 - geo.dVoxelY / 2);
	P.z = -(geo.sVoxelZ / 2 - geo.dVoxelZ / 2);

	//Done for P, now source
	Point3D S;
	S.x = geo.OcrX[i];
	S.y = geo.OcrY[i];
	S.z = geo.OcrZ[i];


	//background center point
	Point3D midPbk, midDbk;

	midPbk.x = (geo.PbkCornerX[i * 4] + geo.PbkCornerX[i * 4 + 3]) / 2;
	midPbk.y = (geo.PbkCornerY[i * 4] + geo.PbkCornerY[i * 4 + 3]) / 2;
	midPbk.z = (geo.PbkCornerZ[i * 4] + geo.PbkCornerZ[i * 4 + 3]) / 2;

	//background point
	Point3D Pfinal, Pfinalu0, Pfinalv0;

	Pfinal.x = geo.PbkCornerX[i * 4];
	Pfinal.y = geo.PbkCornerY[i * 4];
	Pfinal.z = geo.PbkCornerZ[i * 4];

	Pfinalv0.x = (geo.PbkCornerX[i * 4 + 1] - geo.PbkCornerX[i * 4]) / (geo.nCamV[i] - 1);
	Pfinalv0.y = (geo.PbkCornerY[i * 4 + 1] - geo.PbkCornerY[i * 4]) / (geo.nCamV[i] - 1);
	Pfinalv0.z = (geo.PbkCornerZ[i * 4 + 1] - geo.PbkCornerZ[i * 4]) / (geo.nCamV[i] - 1);

	Pfinalu0.x = (geo.PbkCornerX[i * 4 + 2] - geo.PbkCornerX[i * 4]) / (geo.nCamU[i] - 1);
	Pfinalu0.y = (geo.PbkCornerY[i * 4 + 2] - geo.PbkCornerY[i * 4]) / (geo.nCamU[i] - 1);
	Pfinalu0.z = (geo.PbkCornerZ[i * 4 + 2] - geo.PbkCornerZ[i * 4]) / (geo.nCamU[i] - 1);

	// As we want the (0,0,0) to be in a corner of the image, we need to translate everything (after rotation);
	P.x = P.x + geo.sVoxelX / 2;      P.y = P.y + geo.sVoxelY / 2;          P.z = P.z + geo.sVoxelZ / 2;
	S.x = S.x + geo.sVoxelX / 2;      S.y = S.y + geo.sVoxelY / 2;          S.z = S.z + geo.sVoxelZ / 2;
	midPbk.x = midPbk.x + geo.sVoxelX / 2;    midPbk.y = midPbk.y + geo.sVoxelY / 2;    midPbk.z = midPbk.z + geo.sVoxelZ / 2;

	//4. Scale everything so dVoxel==1
	P.x = P.x / geo.dVoxelX;      P.y = P.y / geo.dVoxelY;        P.z = P.z / geo.dVoxelZ;
	S.x = S.x / geo.dVoxelX;      S.y = S.y / geo.dVoxelY;        S.z = S.z / geo.dVoxelZ;

	midPbk.x = midPbk.x / geo.dVoxelX;    midPbk.y = midPbk.y / geo.dVoxelY;    midPbk.z = midPbk.z / geo.dVoxelZ;

	midDbk.x = midPbk.x - S.x;
	midDbk.y = midPbk.y - S.y;
	midDbk.z = midPbk.z - S.z;

	Pfinalu0.x = Pfinalu0.x / geo.dVoxelX;    Pfinalu0.y = Pfinalu0.y / geo.dVoxelY;      Pfinalu0.z = Pfinalu0.z / geo.dVoxelZ;
	Pfinalv0.x = Pfinalv0.x / geo.dVoxelX;    Pfinalv0.y = Pfinalv0.y / geo.dVoxelY;      Pfinalv0.z = Pfinalv0.z / geo.dVoxelZ;



	// return
	*xyzorigin = P;
	*source = S;
	*midPtBK = midPbk;
	*midDirBK = midDbk;

	*deltaU = Pfinalu0;
	*deltaV = Pfinalv0;

}

void computeBackgroundCoef(Geometry geo, int i, Point3D S, Point3D midPtBK, float* DSD, float* co)
{
	float d = sqrt((midPtBK.x - S.x)*(midPtBK.x - S.x) + (midPtBK.y - S.y)*(midPtBK.y - S.y) + (midPtBK.z - S.z)*(midPtBK.z - S.z));
	float M = geo.Zbc[i] / geo.fd[i];
	float c = geo.dVoxelX*geo.dVoxelY*geo.dVoxelZ / geo.dCamU[i] / geo.dCamV[i] / M / M;

	// return
	*DSD = d;
	*co = c;
}
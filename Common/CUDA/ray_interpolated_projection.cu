/*-------------------------------------------------------------------------
 *
 * CUDA functions for linear interpolation intersection based projection
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
#include "ray_interpolated_projection.h"
#include "BOSLAB_common.h"


#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("%s \n",msg);\
                cudaDeviceReset();\
                mexPrintf("BOSLAB:Ax:linear interpolated",cudaGetErrorString(__err));\
        } \
} while (0)



#define MAXTREADS 1024
#define PROJ_PER_BLOCK 8
#define PIXEL_SIZE_BLOCK 8
 /*GEOMETRY DEFINITION
  *
  *                Detector plane, behind
  *            |-----------------------------|
  *            |                             |
  *            |                             |
  *            |                             |
  *            |                             |
  *            |      +--------+             |
  *            |     /        /|             |
  *   A Z      |    /        / |*D           |
  *   |        |   +--------+  |             |
  *   |        |   |        |  |             |
  *   |        |   |     *O |  +             |
  *    --->y   |   |        | /              |
  *  /         |   |        |/               |
  * V X        |   +--------+                |
  *            |-----------------------------|
  *
  *           *S
  *
  *
  *
  *
  *
  **/
void CreateTextureInterp(const GpuIds& gpuids, const float* imagedata, Geometry geo, cudaArray** d_cuArrTex, cudaTextureObject_t *texImage, bool allocate);

__constant__ Point3D projParamsArrayDev[4 * PROJ_PER_BLOCK];  // Dev means it is on device

__global__ void vecAddInPlaceInterp(float *a, float *b, unsigned long  n)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	// Make sure we do not go out of bounds
	if (idx < n)
		a[idx] = a[idx] + b[idx];
}

template<bool linearBeam>
__global__ void kernelPixelDetectorInterp(Geometry geo,
	float* detector,
	int* d_nCamU,
	int* d_nCamV,
	const int currProjSetNumber,
	const int totalNoOfProjections,
	const int diffselect,
	bool EFtracing,
	cudaTextureObject_t tex) {

	unsigned long long u = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long v = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long long projNumber = threadIdx.z;

	if (projNumber >= PROJ_PER_BLOCK)
		return;

	unsigned long indAlpha = currProjSetNumber * PROJ_PER_BLOCK + projNumber;  // This is the ABSOLUTE projection number in the projection array (for a given GPU)	

	if (indAlpha >= totalNoOfProjections)
		return;

	if (u >= d_nCamU[indAlpha] || v >= d_nCamV[indAlpha])
		return;


	size_t idx = (size_t)(u * (unsigned long long)d_nCamV[indAlpha] + v) + projNumber * (unsigned long long)geo.maxnCamV *(unsigned long long)geo.maxnCamU;


	Point3D uvOrigin = projParamsArrayDev[4 * projNumber];  // 6*projNumber because we have 6 Point3D values per projection
	Point3D deltaU = projParamsArrayDev[4 * projNumber + 1];
	Point3D deltaV = projParamsArrayDev[4 * projNumber + 2];
	Point3D source = projParamsArrayDev[4 * projNumber + 3];

	/////// Get coordinates XYZ of pixel UV
	unsigned long pixelV = v;
	unsigned long pixelU = u;


	Point3D P;
	P.x = (uvOrigin.x + pixelU * deltaU.x + pixelV * deltaV.x);
	P.y = (uvOrigin.y + pixelU * deltaU.y + pixelV * deltaV.y);
	P.z = (uvOrigin.z + pixelU * deltaU.z + pixelV * deltaV.z);

	Point3D ray;
	// vector of Xray
	ray.x = P.x - source.x;
	ray.y = P.y - source.y;
	ray.z = P.z - source.z;
	float eps = 0.001;
	ray.x = (fabsf(ray.x) < eps) ? 0 : ray.x;
	ray.y = (fabsf(ray.y) < eps) ? 0 : ray.y;
	ray.z = (fabsf(ray.z) < eps) ? 0 : ray.z;

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
		detector[idx] = 0;
		return;
	}

	// Length is the ray length in normalized space
	Point3D flowIn, flowOut;
	flowIn.x = source.x + am * ray.x;
	flowIn.y = source.y + am * ray.y;
	flowIn.z = source.z + am * ray.z;

	flowOut.x = source.x + aM * ray.x;
	flowOut.y = source.y + aM * ray.y;
	flowOut.z = source.z + aM * ray.z;

	
	//     //Integrate over the line
	float tx, ty, tz;
	float sum = 0;
	float sum_step1, sum_step0, ds;
	float deltalength;


	if (linearBeam) {
		float i;
		float length = __fsqrt_rd((flowOut.x - flowIn.x)*(flowOut.x - flowIn.x) + (flowOut.y - flowIn.y)*(flowOut.y - flowIn.y) + (flowOut.z - flowIn.z)*(flowOut.z - flowIn.z));
		//now legth is an integer of Nsamples that are required on this line
		length = ceilf(__fdividef(length, geo.accuracy));
		//Divide the directional vector by an integer
		float vectX, vectY, vectZ;
		vectX = __fdividef(flowOut.x - flowIn.x, length);
		vectY = __fdividef(flowOut.y - flowIn.y, length);
		vectZ = __fdividef(flowOut.z - flowIn.z, length);

		deltalength = sqrtf((vectX*geo.dVoxelX)*(vectX*geo.dVoxelX) +
			(vectY*geo.dVoxelY)*(vectY*geo.dVoxelY) +
			(vectZ*geo.dVoxelZ)*(vectZ*geo.dVoxelZ));

		ds = sqrtf(vectX*vectX + vectY*vectY + vectZ*vectZ);
		for (i = 0; i <= length; i = i + 1) {
			tx = vectX * i + flowIn.x;
			ty = vectY * i + flowIn.y;
			tz = vectZ * i + flowIn.z;

			switch (diffselect) {// this line is 94% of time.
			case 0:
				sum += tex3D<float>(tex, tx, ty, tz);
				//sum += tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 0.5f);
				break;
			case 1:
				if (tx > 0 && tx < geo.nVoxelX - 1) {
					sum += (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / (2.0f * geo.dVoxelX);
				}
				else if (tx == 0) {
					sum += (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelX;
				}
				else {
					sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / geo.dVoxelX;
				}

				//sum += (tex3D<float>(tex, tx + 1.5f, ty + 0.5f, tz + 0.5f) - tex3D<float>(tex, tx - 0.5f, ty + 0.5f, tz + 0.5f)) / 2 / geo.dVoxelX;
				break;
			case 2:
				if (ty > 0 && ty < geo.nVoxelY - 1) {
					sum += (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / (2.0f * geo.dVoxelY);
				}
				else if (ty == 0) {
					sum += (tex3D<float>(tex, tx , ty + 1, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelY;
				}
				else {
					sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / geo.dVoxelY;
				}
				//sum += (tex3D<float>(tex, tx + 0.5f, ty + 1.5f, tz + 0.5f) - tex3D<float>(tex, tx + 0.5f, ty - 0.5f, tz + 0.5f)) / 2 / geo.dVoxelY;
				break;
			case 3:
				if (tz > 0 && tz < geo.nVoxelZ - 1) {
					sum += (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (tz == 0) {
					sum += (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelZ;
				}
				else {
					sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty, tz - 1)) / geo.dVoxelZ;
				}
				//sum += (tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 1.5f) - tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz - 1.5f)) / 2 / geo.dVoxelZ;
				break;
			case 4:

				sum_step1 = tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 0.5f);
				if (i > 0) {
					sum_step0 = tex3D<float>(tex, tx - vectX + 0.5f, ty - vectY + 0.5f, tz - vectZ + 0.5f);
					sum += (sum_step1 - sum_step0)/ ds;
				}			
				break;
			default:
				break;
			}
		}
		//Length is not actually a length, but the amount of memreads with given accuracy ("samples per voxel")
		detector[idx] = sum * deltalength;
	}
	else {

		float dnx, dny, dnz;
		float Jx, Jy, Jz, Jnorm;
		Jx = ray.x;
		Jy = ray.y;
		Jz = ray.z;
		Jnorm = sqrtf(Jx*Jx + Jy*Jy + Jz*Jz);
		Jx = __fdividef(Jx, Jnorm);
		Jy = __fdividef(Jy, Jnorm);
		Jz = __fdividef(Jz, Jnorm);
		float ds = geo.accuracy;
		deltalength = sqrtf((Jx*geo.dVoxelX)*(Jx*geo.dVoxelX) +
			(Jy*geo.dVoxelY)*(Jy*geo.dVoxelY) +
			(Jz*geo.dVoxelZ)*(Jz*geo.dVoxelZ))*ds;

		bool isborder = false;
		tx = flowIn.x;
		ty = flowIn.y;
		tz = flowIn.z;
		switch (diffselect) {// this line is 94% of time.
		case 0:
			sum += tex3D<float>(tex, tx, ty, tz) * deltalength;
			//sum += tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 0.5f) * deltalength;
			break;
		case 1:
			if (tx > 0 && tx < geo.nVoxelX - 1) {
				sum += (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / (2.0f * geo.dVoxelX) * deltalength;
			}
			else if (tx == 0) {
				sum += (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelX * deltalength;
			}
			else {
				sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / geo.dVoxelX * deltalength;
			}
			//sum += (tex3D<float>(tex, tx + 1.5f, ty + 0.5f, tz + 0.5f) - tex3D<float>(tex, tx - 0.5f, ty + 0.5f, tz + 0.5f)) / 2 / geo.dVoxelX * deltalength;
			break;
		case 2:
			if (ty > 0 && ty < geo.nVoxelY - 1) {
				sum += (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / (2.0f * geo.dVoxelY) * deltalength;
			}
			else if (ty == 0) {
				sum += (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelY * deltalength;
			}
			else {
				sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / geo.dVoxelY * deltalength;
			}
			//sum += (tex3D<float>(tex, tx + 0.5f, ty + 1.5f, tz + 0.5f) - tex3D<float>(tex, tx + 0.5f, ty - 0.5f, tz + 0.5f)) / 2 / geo.dVoxelY * deltalength;
			break;
		case 3:
			if (tz > 0 && tz < geo.nVoxelZ - 1) {
				sum += (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz - 1)) / (2.0f * geo.dVoxelZ) * deltalength;
			}
			else if (tz == 0) {
				sum += (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelZ * deltalength;
			}
			else {
				sum += (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty, tz - 1)) / geo.dVoxelZ * deltalength;
			}
			//sum += (tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 1.5f) - tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz - 0.5f)) / 2 / geo.dVoxelZ * deltalength;
			break;
		default:
			break;
		}
		if (EFtracing) {
			do{
				tx = Jx*ds + tx;
				ty = Jy*ds + ty;
				tz = Jz*ds + tz;
				if (tx > 0 && tx < geo.nVoxelX - 1) {
					dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / (2.0f * geo.dVoxelX);
				}
				else if (tx == 0) {
					dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelX;
				}
				else {
					dnx = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / geo.dVoxelX;
				}

				if (ty > 0 && ty < geo.nVoxelY - 1) {
					dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / (2.0f * geo.dVoxelY);
				}
				else if (ty == 0) {
					dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelY;
				}
				else {
					dny = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / geo.dVoxelY;
				}

				if (tz > 0 && tz < geo.nVoxelZ - 1) {
					dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (tz == 0) {
					dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelZ;
				}
				else {
					dnz = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty, tz - 1)) / geo.dVoxelZ;
				}
				//dnx = (tex3D<float>(tex, tx + 1.5f, ty + 0.5f, tz + 0.5f) - tex3D<float>(tex, tx - 0.5f, ty + 0.5f, tz + 0.5f)) / 2 / geo.dVoxelX;
				//dny = (tex3D<float>(tex, tx + 0.5f, ty + 1.5f, tz + 0.5f) - tex3D<float>(tex, tx + 0.5f, ty - 0.5f, tz + 0.5f)) / 2 / geo.dVoxelY;
				//dnz = (tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 1.5f) - tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz - 0.5f)) / 2 / geo.dVoxelZ;

				Jx = Jx + dnx * ds ;
				Jy = Jy + dny * ds ;
				Jz = Jz + dnz * ds ;

				Jnorm = sqrtf(Jx*Jx + Jy * Jy + Jz * Jz);
				Jx = __fdividef(Jx, Jnorm);
				Jy = __fdividef(Jy, Jnorm);
				Jz = __fdividef(Jz, Jnorm);

				if (tx < 0 | ty < 0 | tz < 0 | tx>geo.nVoxelX | ty>geo.nVoxelY | tz >geo.nVoxelZ) {
					isborder = true;
				}		

				deltalength = sqrtf((Jx*geo.dVoxelX)*(Jx*geo.dVoxelX) + (Jy*geo.dVoxelY)*(Jy*geo.dVoxelY) + (Jz*geo.dVoxelZ)*(Jz*geo.dVoxelZ))*ds;
				switch (diffselect) {// this line is 94% of time.
				case 0:
					sum += tex3D<float>(tex, tx, ty, tz) * deltalength;
					//sum += tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 0.5f) * deltalength;
					break;
				case 1:
					sum += dnx * deltalength;
					break;
				case 2:
					sum += dny * deltalength;
					break;
				case 3:
					sum += dnz * deltalength;
					break;
				default:
					break;
				}
			} while (!isborder);
			detector[idx] = sum;
		}
		else {
			float cx, cy, cz;
			Point3D k1, k2, k3, k4;
			if (tx > 0 && tx < geo.nVoxelX - 1) {
				dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / (2.0f * geo.dVoxelX);
			}
			else if (tx == 0) {
				dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelX;
			}
			else {
				dnx = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / geo.dVoxelX;
			}

			if (ty > 0 && ty < geo.nVoxelY - 1) {
				dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / (2.0f * geo.dVoxelY);
			}
			else if (ty == 0) {
				dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelY;
			}
			else {
				dny = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / geo.dVoxelY;
			}

			if (tz > 0 && tz < geo.nVoxelZ - 1) {
				dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz - 1)) / (2.0f * geo.dVoxelZ);
			}
			else if (tz == 0) {
				dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelZ;
			}
			else {
				dnz = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty, tz - 1)) / geo.dVoxelZ;
			}
			//dnx = (tex3D<float>(tex, tx + 1.5f, ty + 0.5f, tz + 0.5f) - tex3D<float>(tex, tx - 0.5f, ty + 0.5f, tz + 0.5f)) / 2 / geo.dVoxelX;
			//dny = (tex3D<float>(tex, tx + 0.5f, ty + 1.5f, tz + 0.5f) - tex3D<float>(tex, tx + 0.5f, ty - 0.5f, tz + 0.5f)) / 2 / geo.dVoxelY;
			//dnz = (tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 1.5f) - tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz - 0.5f)) / 2 / geo.dVoxelZ;
			do {
				

				k1.x = ds * dnx;
				k1.y = ds * dny;
				k1.z = ds * dnz;

				cx = tx + ds / 2 * Jx + ds / 8 * k1.x;
				cy = ty + ds / 2 * Jy + ds / 8 * k1.y;
				cz = tz + ds / 2 * Jz + ds / 8 * k1.z;

				if (cx > 0 && cx < geo.nVoxelX - 1) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / (2.0f * geo.dVoxelX);
				}
				else if (cx == 0) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelX;
				}
				else {
					dnx = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / geo.dVoxelX;
				}

				if (cy > 0 && cy < geo.nVoxelY - 1) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / (2.0f * geo.dVoxelY);
				}
				else if (cy == 0) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelY;
				}
				else {
					dny = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / geo.dVoxelY;
				}

				if (cz > 0 && cz < geo.nVoxelZ - 1) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (cz == 0) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelZ;
				}
				else {
					dnz = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy, cz - 1)) / geo.dVoxelZ;
				}

				//dnx = (tex3D<float>(tex, cx + 1.5f, cy + 0.5f, cz + 0.5f) - tex3D<float>(tex, cx - 0.5f, cy + 0.5f, cz + 0.5f)) / 2 / geo.dVoxelX;
				//dny = (tex3D<float>(tex, cx + 0.5f, cy + 1.5f, cz + 0.5f) - tex3D<float>(tex, cx + 0.5f, cy - 0.5f, cz + 0.5f)) / 2 / geo.dVoxelY;
				//dnz = (tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz + 1.5f) - tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz - 0.5f)) / 2 / geo.dVoxelZ;

				k2.x = ds * dnx;
				k2.y = ds * dny;
				k2.z = ds * dnz;

				cx = tx + ds / 2 * Jx + ds / 8 * k2.x;
				cy = ty + ds / 2 * Jy + ds / 8 * k2.y;
				cz = tz + ds / 2 * Jz + ds / 8 * k2.z;


				if (cx > 0 && cx < geo.nVoxelX - 1) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / (2.0f * geo.dVoxelX);
				}
				else if (cx == 0) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelX;
				}
				else {
					dnx = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / geo.dVoxelX;
				}

				if (cy > 0 && cy < geo.nVoxelY - 1) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / (2.0f * geo.dVoxelY);
				}
				else if (cy == 0) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelY;
				}
				else {
					dny = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / geo.dVoxelY;
				}

				if (cz > 0 && cz < geo.nVoxelZ - 1) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (cz == 0) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelZ;
				}
				else {
					dnz = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy, cz - 1)) / geo.dVoxelZ;
				}

				//dnx = (tex3D<float>(tex, cx + 1.5f, cy + 0.5f, cz + 0.5f) - tex3D<float>(tex, cx - 0.5f, cy + 0.5f, cz + 0.5f)) / 2 / geo.dVoxelX;
				//dny = (tex3D<float>(tex, cx + 0.5f, cy + 1.5f, cz + 0.5f) - tex3D<float>(tex, cx + 0.5f, cy - 0.5f, cz + 0.5f)) / 2 / geo.dVoxelY;
				//dnz = (tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz + 1.5f) - tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz - 0.5f)) / 2 / geo.dVoxelZ;

				k3.x = ds * dnx;
				k3.y = ds * dny;
				k3.z = ds * dnz;

				cx = tx + ds * Jx + ds / 2 * k3.x;
				cy = ty + ds * Jy + ds / 2 * k3.y;
				cz = tz + ds * Jz + ds / 2 * k3.z;

				if (cx > 0 && cx < geo.nVoxelX - 1) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / (2.0f * geo.dVoxelX);
				}
				else if (cx == 0) {
					dnx = (tex3D<float>(tex, cx + 1, cy, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelX;
				}
				else {
					dnx = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx - 1, cy, cz)) / geo.dVoxelX;
				}

				if (cy > 0 && cy < geo.nVoxelY - 1) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / (2.0f * geo.dVoxelY);
				}
				else if (cy == 0) {
					dny = (tex3D<float>(tex, cx, cy + 1, cz) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelY;
				}
				else {
					dny = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy - 1, cz)) / geo.dVoxelY;
				}

				if (cz > 0 && cz < geo.nVoxelZ - 1) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (cz == 0) {
					dnz = (tex3D<float>(tex, cx, cy, cz + 1) - tex3D<float>(tex, cx, cy, cz)) / geo.dVoxelZ;
				}
				else {
					dnz = (tex3D<float>(tex, cx, cy, cz) - tex3D<float>(tex, cx, cy, cz - 1)) / geo.dVoxelZ;
				}

				//dnx = (tex3D<float>(tex, cx + 1.5f, cy + 0.5f, cz + 0.5f) - tex3D<float>(tex, cx - 0.5f, cy + 0.5f, cz + 0.5f)) / 2 / geo.dVoxelX;
				//dny = (tex3D<float>(tex, cx + 0.5f, cy + 1.5f, cz + 0.5f) - tex3D<float>(tex, cx + 0.5f, cy - 0.5f, cz + 0.5f)) / 2 / geo.dVoxelY;
				//dnz = (tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz + 1.5f) - tex3D<float>(tex, cx + 0.5f, cy + 0.5f, cz - 0.5f)) / 2 / geo.dVoxelZ;

				k4.x = ds * dnx;
				k4.y = ds * dny;
				k4.z = ds * dnz;

				tx = tx + ds / 6 * (6 * Jx + k1.x + k2.x + k3.x);
				ty = ty + ds / 6 * (6 * Jy + k1.y + k2.y + k3.y);
				tz = tz + ds / 6 * (6 * Jz + k1.z + k2.z + k3.z);

				Jx = Jx + 1 / 6 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x);
				Jy = Jy + 1 / 6 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
				Jz = Jz + 1 / 6 * (k1.z + 2 * k2.z + 2 * k3.z + k4.z);

				Jnorm = sqrtf(Jx*Jx + Jy * Jy + Jz * Jz);
				Jx = __fdividef(Jx, Jnorm);
				Jy = __fdividef(Jy, Jnorm);
				Jz = __fdividef(Jz, Jnorm);

				if (tx > 0 && tx < geo.nVoxelX - 1) {
					dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / (2.0f * geo.dVoxelX);
				}
				else if (tx == 0) {
					dnx = (tex3D<float>(tex, tx + 1, ty, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelX;
				}
				else {
					dnx = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx - 1, ty, tz)) / geo.dVoxelX;
				}

				if (ty > 0 && ty < geo.nVoxelY - 1) {
					dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / (2.0f * geo.dVoxelY);
				}
				else if (ty == 0) {
					dny = (tex3D<float>(tex, tx, ty + 1, tz) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelY;
				}
				else {
					dny = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty - 1, tz)) / geo.dVoxelY;
				}

				if (tz > 0 && tz < geo.nVoxelZ - 1) {
					dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz - 1)) / (2.0f * geo.dVoxelZ);
				}
				else if (tz == 0) {
					dnz = (tex3D<float>(tex, tx, ty, tz + 1) - tex3D<float>(tex, tx, ty, tz)) / geo.dVoxelZ;
				}
				else {
					dnz = (tex3D<float>(tex, tx, ty, tz) - tex3D<float>(tex, tx, ty, tz - 1)) / geo.dVoxelZ;
				}
				//dnx = (tex3D<float>(tex, tx + 1.5f, ty + 0.5f, tz + 0.5f) - tex3D<float>(tex, tx - 0.5f, ty + 0.5f, tz + 0.5f)) / 2 / geo.dVoxelX;
				//dny = (tex3D<float>(tex, tx + 0.5f, ty + 1.5f, tz + 0.5f) - tex3D<float>(tex, tx + 0.5f, ty - 0.5f, tz + 0.5f)) / 2 / geo.dVoxelY;
				//dnz = (tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 1.5f) - tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz - 0.5f)) / 2 / geo.dVoxelZ;

				if (tx < 0 | ty < 0 | tz < 0 | tx>geo.nVoxelX | ty>geo.nVoxelY | tz >geo.nVoxelZ ) {
					isborder = true;
				}

				deltalength = sqrtf((Jx*geo.dVoxelX)*(Jx*geo.dVoxelX) + (Jy*geo.dVoxelY)*(Jy*geo.dVoxelY) + (Jz*geo.dVoxelZ)*(Jz*geo.dVoxelZ))*ds;

				switch (diffselect) {// this line is 94% of time.
				case 0:
					sum += tex3D<float>(tex, tx, ty , tz ) * deltalength;
					//sum += tex3D<float>(tex, tx + 0.5f, ty + 0.5f, tz + 0.5f) * deltalength;
					break;
				case 1:
					sum += dnx * deltalength;
					break;
				case 2:
					sum += dny * deltalength;
					break;
				case 3:
					sum += dnz * deltalength;
					break;
				default:
					break;
				}

			} while (!isborder);
			detector[idx] = sum;
		}
	}

	
}



// legnth(angles)=3 x nagnles, as we have roll, pitch, yaw.
int interpolation_projection(float  *  img, Geometry geo, const int diffselect, bool linearBeam, bool EFtracing, float** result, const GpuIds& gpuids, Blur blur) {


	// Prepare for MultiGPU
	int deviceCount = gpuids.GetLength();
	cudaCheckErrors("Device query fail");
	if (deviceCount == 0) {
		mexPrintf("Ax:Interpolated_projection:GPUselect", "There are no available device(s) that support CUDA\n");
	}
	//
	// CODE assumes
	// 1.-All available devices are usable by this code
	// 2.-All available devices are equal, they are the same machine (warning thrown)
	// Check the available devices, and if they are the same
	if (!gpuids.AreEqualDevices()) {
		mexPrintf("Ax:Interpolated_projection:GPUselect", "Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
	}
	int dev;

	// Check free memory
	size_t mem_GPU_global;
	checkFreeMemory(gpuids, &mem_GPU_global);

	// printf("geo.nDetec (U, V) = %d, %d\n", geo.nDetecU, geo.nDetecV);

	size_t mem_image = (unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY*(unsigned long long)geo.nVoxelZ * sizeof(float);
	size_t mem_proj = (unsigned long long)geo.maxnCamU*(unsigned long long)geo.maxnCamV * sizeof(float);

	// Does everything fit in the GPUs?
	const bool fits_in_memory = mem_image + 2 * PROJ_PER_BLOCK*mem_proj < mem_GPU_global;
	unsigned int splits = 1;
	if (!fits_in_memory) {
		// Nope nope.
		// approx free memory we have. We already have left some extra 5% free for internal stuff
		// we need a second projection memory to combine multi-GPU stuff.
		size_t mem_free = mem_GPU_global - 4 * PROJ_PER_BLOCK*mem_proj;
		splits = mem_image / mem_free + 1;// Ceil of the truncation
	}
	Geometry* geoArray = (Geometry*)malloc(splits * sizeof(Geometry));
	splitImageInterp(splits, geo, geoArray);


	// Allocate auiliary memory for projections on the GPU to accumulate partial results
	float ** dProjection_accum;
	size_t num_bytes_proj = PROJ_PER_BLOCK * geo.maxnCamU*geo.maxnCamV * sizeof(float);
	if (!fits_in_memory) {
		dProjection_accum = (float**)malloc(2 * deviceCount * sizeof(float*));
		for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			for (int i = 0; i < 2; ++i) {
				cudaMalloc((void**)&dProjection_accum[dev * 2 + i], num_bytes_proj);
				cudaMemset(dProjection_accum[dev * 2 + i], 0, num_bytes_proj);
				cudaCheckErrors("cudaMallocauxiliarty projections fail");
			}
		}
	}

	// This is happening regarthless if the image fits on memory
	float** dProjection = (float**)malloc(2 * deviceCount * sizeof(float*));
	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);

		for (int i = 0; i < 2; ++i) {
			cudaMalloc((void**)&dProjection[dev * 2 + i], num_bytes_proj);
			cudaMemset(dProjection[dev * 2 + i], 0, num_bytes_proj);
			cudaCheckErrors("cudaMalloc projections fail");
		}
	}




	//Pagelock memory for synchronous copy.
	// Lets try to make the host memory pinned:
	// We laredy queried the GPU and assuemd they are the same, thus should have the same attributes.
	int isHostRegisterSupported = 0;
#if CUDART_VERSION >= 9020
	cudaDeviceGetAttribute(&isHostRegisterSupported, cudaDevAttrHostRegisterSupported, gpuids[0]);
#endif
	// empirical testing shows that when the image split is smaller than 1 (also implies the image is not very big), the time to
	// pin the memory is greater than the lost time in Synchronously launching the memcpys. This is only worth it when the image is too big.

#ifndef NO_PINNED_MEMORY
	if (isHostRegisterSupported & splits > 1) {
		cudaHostRegister(img, (size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)geo.nVoxelZ*(size_t)sizeof(float), cudaHostRegisterPortable);
	}
	cudaCheckErrors("Error pinning memory");
#endif

	// allocate geo.nCamU and geo.nCamV into device
	int* d_nCamU;
	int* d_nCamV;
	cudaMalloc((void**)&d_nCamU, geo.numCam * sizeof(int));
	cudaMalloc((void**)&d_nCamV, geo.numCam * sizeof(int));

	cudaMemcpy(d_nCamU, geo.nCamU, geo.numCam * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nCamV, geo.nCamV, geo.numCam * sizeof(int), cudaMemcpyHostToDevice);

	// auxiliary variables
	Point3D source, deltaU, deltaV, uvOrigin;
	Point3D* projParamsArrayHost = 0;
	cudaMallocHost((void**)&projParamsArrayHost, 4 * PROJ_PER_BLOCK * sizeof(Point3D));
	float* projFloatsArrayHost = 0;
	cudaMallocHost((void**)&projFloatsArrayHost, 2 * PROJ_PER_BLOCK * sizeof(float));
	cudaCheckErrors("Error allocating auxiliary constant memory");

	// Create Streams for overlapping memcopy and compute
	int nStream_device = 2;
	int nStreams = deviceCount * nStream_device;
	cudaStream_t* stream = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));

	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		for (int i = 0; i < nStream_device; ++i) {
			cudaStreamCreate(&stream[i + dev * nStream_device]);

		}
	}
	cudaCheckErrors("Stream creation fail");
	int nangles_device = (geo.numCam + deviceCount - 1) / deviceCount;
	int nangles_last_device = (geo.numCam - (deviceCount - 1)*nangles_device);
	unsigned int noOfKernelCalls = (nangles_device + PROJ_PER_BLOCK - 1) / PROJ_PER_BLOCK;  // We'll take care of bounds checking inside the loop if nalpha is not divisible by PROJ_PER_BLOCK
	unsigned int noOfKernelCallsLastDev = (nangles_last_device + PROJ_PER_BLOCK - 1) / PROJ_PER_BLOCK; // we will use this in the memory management.
	int projection_this_block;



	cudaTextureObject_t *texImg = new cudaTextureObject_t[deviceCount];
	cudaArray **d_cuArrTex = new cudaArray*[deviceCount];
	for (unsigned int sp = 0; sp < splits; sp++) {
		// Create texture objects for all GPUs


		size_t linear_idx_start;
		// They are all the same size, except the last one.
		linear_idx_start = (size_t)sp*(size_t)geoArray[0].nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)geoArray[0].nVoxelZ;
		CreateTextureInterp(gpuids, &img[linear_idx_start], geoArray[sp], d_cuArrTex, texImg, !sp);
		cudaCheckErrors("Texture object creation fail");


		int divU, divV;
		divU = PIXEL_SIZE_BLOCK;
		divV = PIXEL_SIZE_BLOCK;
		dim3 grid((geoArray[sp].maxnCamU + divU - 1) / divU, (geoArray[0].maxnCamV + divV - 1) / divV, 1);
		dim3 block(divU, divV, PROJ_PER_BLOCK);

		unsigned int proj_global;
		float maxdist;
		// Now that we have prepared the image (piece of image) and parameters for kernels
		// we project for all angles.
		for (unsigned int i = 0; i < noOfKernelCalls; i++) {
			for (dev = 0; dev < deviceCount; dev++) {
				cudaSetDevice(gpuids[dev]);

				for (unsigned int j = 0; j < PROJ_PER_BLOCK; j++) {
					proj_global = (i*PROJ_PER_BLOCK + j) + dev * nangles_device;
					if (proj_global >= geo.numCam)
						break;
					if ((i*PROJ_PER_BLOCK + j) >= nangles_device)
						break;

					//Precompute per angle constant stuff for speed
					computeDeltas(geoArray[sp], proj_global, &uvOrigin, &deltaU, &deltaV, &source, blur);
					//Ray tracing!
					projParamsArrayHost[4 * j] = uvOrigin;		// 6*j because we have 6 Point3D values per projection
					projParamsArrayHost[4 * j + 1] = deltaU;
					projParamsArrayHost[4 * j + 2] = deltaV;
					projParamsArrayHost[4 * j + 3] = source;

				}

				cudaMemcpyToSymbolAsync(projParamsArrayDev, projParamsArrayHost, sizeof(Point3D) * 4 * PROJ_PER_BLOCK, 0, cudaMemcpyHostToDevice, stream[dev*nStream_device]);
				cudaStreamSynchronize(stream[dev*nStream_device]);
				cudaCheckErrors("kernel fail");
				if (linearBeam) {
					kernelPixelDetectorInterp<true> << <grid, block, 0, stream[dev*nStream_device] >> > (geoArray[sp], dProjection[(i % 2) + dev * 2], d_nCamU, d_nCamV, i, nangles_device, diffselect, EFtracing, texImg[dev]);
				}
				else {
					kernelPixelDetectorInterp<false> << <grid, block, 0, stream[dev*nStream_device] >> > (geoArray[sp], dProjection[(i % 2) + dev * 2], d_nCamU, d_nCamV, i, nangles_device, diffselect, EFtracing, texImg[dev]);
				}

			}


			// Now that the computation is happening, we need to either prepare the memory for
			// combining of the projections (splits>1) and start removing previous results.


			// If our image does not fit in memory then we need to make sure we accumulate previous results too.
			// This is done in 2 steps: 
			// 1)copy previous results back into GPU 
			// 2)accumulate with current results
			// The code to take them out is the same as when there are no splits needed
			if (!fits_in_memory&&sp > 0)
			{
				// 1) grab previous results and put them in the auxiliary variable dProjection_accum
				for (dev = 0; dev < deviceCount; dev++)
				{
					cudaSetDevice(gpuids[dev]);
					//Global index of FIRST projection on this set on this GPU
					proj_global = i * PROJ_PER_BLOCK + dev * nangles_device;
					if (proj_global >= geo.numCam)
						break;

					// Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
					if (i + 1 == noOfKernelCalls) //is it the last block?
						projection_this_block = min(nangles_device - (noOfKernelCalls - 1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
							geo.numCam - proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
					else
						projection_this_block = PROJ_PER_BLOCK;
					cudaMemcpyAsync(dProjection_accum[(i % 2) + dev * 2], result[proj_global], projection_this_block*geo.maxnCamV*geo.maxnCamU * sizeof(float), cudaMemcpyHostToDevice, stream[dev * 2 + 1]);
				}
				//  2) take the results from current compute call and add it to the code in execution.
				for (dev = 0; dev < deviceCount; dev++)
				{
					cudaSetDevice(gpuids[dev]);
					//Global index of FIRST projection on this set on this GPU
					proj_global = i * PROJ_PER_BLOCK + dev * nangles_device;
					if (proj_global >= geo.numCam)
						break;

					// Unless its the last projection set, we have PROJ_PER_BLOCK angles. Otherwise...
					if (i + 1 == noOfKernelCalls) //is it the last block?
						projection_this_block = min(nangles_device - (noOfKernelCalls - 1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
							geo.numCam - proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)
					else
						projection_this_block = PROJ_PER_BLOCK;
					cudaStreamSynchronize(stream[dev * 2 + 1]); // wait until copy is finished
					vecAddInPlaceInterp << <(geo.maxnCamU*geo.maxnCamV*projection_this_block + MAXTREADS - 1) / MAXTREADS, MAXTREADS, 0, stream[dev * 2] >> > (dProjection[(i % 2) + dev * 2], dProjection_accum[(i % 2) + dev * 2], (unsigned long)geo.maxnCamU*geo.maxnCamV*projection_this_block);
				}
			} // end accumulation case, where the image needs to be split 

			// Now, lets get out the projections from the previous execution of the kernels.
			if (i > 0)
			{
				for (dev = 0; dev < deviceCount; dev++)
				{
					cudaSetDevice(gpuids[dev]);
					//Global index of FIRST projection on previous set on this GPU
					proj_global = (i - 1)*PROJ_PER_BLOCK + dev * nangles_device;
					if (dev + 1 == deviceCount) {    //is it the last device?
						// projections assigned to this device is >=nangles_device-(deviceCount-1) and < nangles_device
						if (i - 1 < noOfKernelCallsLastDev) {
							// The previous set(block) was not empty.
							projection_this_block = min(PROJ_PER_BLOCK, geo.numCam - proj_global);
						}
						else {
							// The previous set was empty.
							// This happens if deviceCount > PROJ_PER_BLOCK+1.
							// e.g. PROJ_PER_BLOCK = 9, deviceCount = 11, nangles = 199.
							// e.g. PROJ_PER_BLOCK = 1, deviceCount =  3, nangles =   7.
							break;
						}
					}
					else {
						projection_this_block = PROJ_PER_BLOCK;
					}
					cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(i % 2)) + dev * 2], projection_this_block*geo.maxnCamV*geo.maxnCamU * sizeof(float), cudaMemcpyDeviceToHost, stream[dev * 2 + 1]);
				}
			}
			// Make sure Computation on kernels has finished before we launch the next batch.
			for (dev = 0; dev < deviceCount; dev++)
			{
				cudaSetDevice(gpuids[dev]);
				cudaStreamSynchronize(stream[dev * 2]);
			}
		} // End noOfKernelCalls (i) loop.

		// We still have the last set of projections to get out of GPUs
		for (dev = 0; dev < deviceCount; dev++)
		{
			cudaSetDevice(gpuids[dev]);
			//Global index of FIRST projection on this set on this GPU
			proj_global = (noOfKernelCalls - 1)*PROJ_PER_BLOCK + dev * nangles_device;
			if (proj_global >= geo.numCam)
				break;
			// How many projections are left here?
			projection_this_block = min(nangles_device - (noOfKernelCalls - 1)*PROJ_PER_BLOCK, //the remaining angles that this GPU had to do (almost never PROJ_PER_BLOCK)
				geo.numCam - proj_global);                              //or whichever amount is left to finish all (this is for the last GPU)

			cudaDeviceSynchronize(); //Not really necesary, but just in case, we los nothing. 
			cudaCheckErrors("Error at copying the last set of projections out (or in the previous copy)");
			cudaMemcpyAsync(result[proj_global], dProjection[(int)(!(noOfKernelCalls % 2)) + dev * 2], projection_this_block*geo.maxnCamV*geo.maxnCamU * sizeof(float), cudaMemcpyDeviceToHost, stream[dev * 2 + 1]);
		}
		// Make sure everyone has done their bussiness before the next image split:
		for (dev = 0; dev < deviceCount; dev++)
		{
			cudaSetDevice(gpuids[dev]);
			cudaDeviceSynchronize();
		}
	} // End image split loop.

	cudaCheckErrors("Main loop  fail");
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaDestroyTextureObject(texImg[dev]);
		cudaFreeArray(d_cuArrTex[dev]);
	}
	delete[] texImg; texImg = 0;
	delete[] d_cuArrTex; d_cuArrTex = 0;
	// Freeing Stage
	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaFree(dProjection[dev * 2]);
		cudaFree(dProjection[dev * 2 + 1]);

	}
	free(dProjection);

	if (!fits_in_memory) {
		for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			cudaFree(dProjection_accum[dev * 2]);
			cudaFree(dProjection_accum[dev * 2 + 1]);

		}
		free(dProjection_accum);
	}
	freeGeoArray(splits, geoArray);
	cudaFreeHost(projParamsArrayHost);
	cudaFreeHost(projFloatsArrayHost);


	for (int i = 0; i < nStreams; ++i)
		cudaStreamDestroy(stream[i]);
#ifndef NO_PINNED_MEMORY
	if (isHostRegisterSupported & splits > 1) {
		cudaHostUnregister(img);
	}
#endif
	cudaCheckErrors("cudaFree  fail");

	//     cudaDeviceReset();
	return 0;
}
void CreateTextureInterp(const GpuIds& gpuids, const float* imagedata, Geometry geo, cudaArray** d_cuArrTex, cudaTextureObject_t *texImage, bool allocate)
{
	const unsigned int num_devices = gpuids.GetLength();
	//size_t size_image=geo.nVoxelX*geo.nVoxelY*geo.nVoxelZ;
	const cudaExtent extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
	if (allocate) {

		for (unsigned int dev = 0; dev < num_devices; dev++) {
			cudaSetDevice(gpuids[dev]);

			//cudaArray Descriptor

			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
			//cuda Array
			cudaMalloc3DArray(&d_cuArrTex[dev], &channelDesc, extent);
			cudaCheckErrors("Texture memory allocation fail");
		}

	}
	for (unsigned int dev = 0; dev < num_devices; dev++) {
		cudaMemcpy3DParms copyParams = { 0 };
		cudaSetDevice(gpuids[dev]);
		//Array creation
		copyParams.srcPtr = make_cudaPitchedPtr((void *)imagedata, extent.width * sizeof(float), extent.width, extent.height);
		copyParams.dstArray = d_cuArrTex[dev];
		copyParams.extent = extent;
		copyParams.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3DAsync(&copyParams);
		//cudaCheckErrors("Texture memory data copy fail");
		//Array creation End
	}
	for (unsigned int dev = 0; dev < num_devices; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaResourceDesc    texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArrTex[dev];
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = false;
		if (geo.accuracy > 1) {
			texDescr.filterMode = cudaFilterModePoint;
			geo.accuracy = 1;
		}
		else {
			texDescr.filterMode = cudaFilterModeLinear;
		}
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp; //cudaAddressModeBorder
		texDescr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject(&texImage[dev], &texRes, &texDescr, NULL);
		cudaCheckErrors("Texture object creation fail");
	}
}

/* This code generates the geometries needed to split the image properly in
 * cases where the entire image does not fit in the memory of the GPU
 **/
void splitImageInterp(unsigned int splits, Geometry geo, Geometry* geoArray) {

	unsigned long splitsize = (geo.nVoxelZ + splits - 1) / splits;// ceil if not divisible
	for (unsigned int sp = 0; sp < splits; sp++) {
		geoArray[sp] = geo;
		// All of them are splitsize, but the last one, possible
		geoArray[sp].nVoxelZ = ((sp + 1)*splitsize < geo.nVoxelZ) ? splitsize : geo.nVoxelZ - splitsize * sp;
		geoArray[sp].sVoxelZ = geoArray[sp].nVoxelZ* geoArray[sp].dVoxelZ;
	}
}



/* This code precomputes The location of the source and the Delta U and delta V (in the warped space)
 * to compute the locations of the x-rays. While it seems verbose and overly-optimized,
 * it does saves about 30% of each of the kernel calls. Thats something!
 **/
void computeDeltas(Geometry geo, unsigned int i, Point3D* uvorigin, Point3D* deltaU, Point3D* deltaV, Point3D* source, Blur blur) {
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

	if (blur.flag == 1) {
		Pfinal.x = geo.PbkCornerX[i * 4] + blur.DeltaX[i * blur.num];
		Pfinal.y = geo.PbkCornerY[i * 4] + blur.DeltaY[i * blur.num];
		Pfinal.z = geo.PbkCornerZ[i * 4] + blur.DeltaZ[i * blur.num];
	}
	else {
		Pfinal.x = geo.PbkCornerX[i * 4];
		Pfinal.y = geo.PbkCornerY[i * 4];
		Pfinal.z = geo.PbkCornerZ[i * 4];
	}

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

#include "mat.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "BOSLAB_common.h"
#include "imgtodisplacement.h"
#include "reconstruction.h"
#include "gpu_buffer_pool.h"

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
				mexPrintf("Error: %s\n", msg);\
				mexPrintf("imgtoreconstruction: %s\n", cudaGetErrorString(__err)); \
        } \
} while (0)

// ------------------------- device utils ------------------------------
__device__ void matmul3x3(const float* a, const float* b, float* c) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

__device__ void transpose3x3(const float* a, float* at) {
	at[0] = a[0]; at[1] = a[3]; at[2] = a[6];
	at[3] = a[1]; at[4] = a[4]; at[5] = a[7];
	at[6] = a[2]; at[7] = a[5]; at[8] = a[8];
}

__device__ void inv3x3(const float* a, float* inva) {
	float det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
	if (det == 0) return;
	float invdet = 1.0f / det;
	inva[0] = (a[4] * a[8] - a[5] * a[7])*invdet;
	inva[1] = (a[2] * a[7] - a[1] * a[8])*invdet;
	inva[2] = (a[1] * a[5] - a[2] * a[4])*invdet;
	inva[3] = (a[5] * a[6] - a[3] * a[8])*invdet;
	inva[4] = (a[0] * a[8] - a[2] * a[6])*invdet;
	inva[5] = (a[2] * a[3] - a[0] * a[5])*invdet;
	inva[6] = (a[3] * a[7] - a[4] * a[6])*invdet;
	inva[7] = (a[1] * a[6] - a[0] * a[7])*invdet;
	inva[8] = (a[0] * a[4] - a[1] * a[3])*invdet;
}

// ------------------------- global utils ------------------------------
// ------------------------- optical flow 
__global__ void writeFlowKernel(
	const float* u,   // size H*W, row-major
	const float* v,
	float* Uout,      // MATLAB layout
	float* Vout,
	int V, int U,
	int k
) {
	int u_idx = blockIdx.x * blockDim.x + threadIdx.x; // x
	int v_idx = blockIdx.y * blockDim.y + threadIdx.y; // y

	if (u_idx >= U || v_idx >= V) return;

	int idx_cv = v_idx * U + u_idx;  // OpenCV row-major
	int idx_ml = v_idx
		+ u_idx * V
		+ k * (U * V);        // MATLAB column-major

	Uout[idx_ml] = u[idx_cv];
	Vout[idx_ml] = v[idx_cv];
}

__global__ void precompute_KRinv_kernel(
	const float* imcam,
	const float* rrcam,
	const float* zbc,
	const float* zpc,
	float* kr_inv_out,
	float* scale_out,
	int numcam)
{
	int camidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (camidx >= numcam) return;

	float k[9], r[9], kt[9], rt[9], kr[9], kr_inv[9];
#pragma unroll
	for (int i = 0; i < 9; i++) {
		k[i] = imcam[camidx * 9 + i];
		r[i] = rrcam[camidx * 9 + i];
	}

	transpose3x3(k, kt);
	transpose3x3(r, rt);
	matmul3x3(kt, rt, kr);
	inv3x3(kr, kr_inv);

#pragma unroll
	for (int i = 0; i < 9; i++)
		kr_inv_out[camidx * 9 + i] = kr_inv[i];

	float zb = zbc[camidx];
	float zp = zpc[camidx];
	scale_out[camidx] = zb / (zb - zp);
}

__global__ void uvtoeps_kernel_opt(
	const float* projectionsu,
	const float* projectionsv,
	float* projectionsob,
	const float* kr_inv,
	const float* scale,
	int numcam, int rows, int cols)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int camidx = blockIdx.z;
	if (row >= rows || col >= cols || camidx >= numcam) return;

	// shared memory for per-block camera
	__shared__ float s_krinv[9];
	__shared__ float s_scale;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
		for (int i = 0; i < 9; i++) s_krinv[i] = kr_inv[camidx * 9 + i];
		s_scale = scale[camidx];
	}
	__syncthreads();

	int pixelindex = camidx * rows*cols + row * cols + col;
	float u = -projectionsu[pixelindex];
	float v = -projectionsv[pixelindex];

	float eps0 = (s_krinv[0] * u + s_krinv[1] * v) * s_scale;
	float eps1 = (s_krinv[3] * u + s_krinv[4] * v) * s_scale;
	float eps2 = (s_krinv[6] * u + s_krinv[7] * v) * s_scale;

	int stride = rows * cols*numcam;
	projectionsob[pixelindex] = eps0;
	projectionsob[pixelindex + stride] = eps1;
	projectionsob[pixelindex + 2 * stride] = eps2;
}
// ------------------------- IRN-TV-CGLS
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
			if (niter_break != 0 && ((iter) % niter_break) != 1 && (d_resL2[iter - 1] > d_resL2[iter - 2])) {
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

void img_to_reconstruction(
	const Geometry& geo,
	const std::vector<cv::Mat>& iref,
	const std::vector<cv::Mat>& idis,
	const GpuIds& gpuids,
	float* result,
	const ReconstructionPara& reconP,
	GPUBufferPool& gpuPool)
{

	cudaSetDevice(0);

	// ================= timing events =================
	cudaEvent_t ev_img_start, ev_img_end;
	cudaEvent_t ev_rec_start, ev_rec_end;

	cudaEventCreate(&ev_img_start);
	cudaEventCreate(&ev_img_end);
	cudaEventCreate(&ev_rec_start);
	cudaEventCreate(&ev_rec_end);

	// ---------------- Farneback ----------------
	auto farn = cv::cuda::FarnebackOpticalFlow::create(
		3, 0.5, false, 16, 5, 5, 1.1, 0);

	// ---------------- PRE-ALLOCATE GpuMat ----------------
	dim3 block(16, 16);
	dim3 grid((geo.maxnCamU + 15) / 16, (geo.maxnCamV + 15) / 16);

	
	// =====================================================
	// ============== img -> displacement ==================
	// =====================================================
	cudaEventRecord(ev_img_start, 0);

	for (int k = 0; k < geo.numCam; ++k)
	{
		gpuPool.d0[k].upload(iref[k], gpuPool.cvStreams[k]);
		gpuPool.d1[k].upload(idis[k], gpuPool.cvStreams[k]);

		farn->calc(gpuPool.d0[k], gpuPool.d1[k], gpuPool.d_flow[k], gpuPool.cvStreams[k]);

		cv::cuda::GpuMat planes[2] = { gpuPool.d_u[k], gpuPool.d_v[k] };
		cv::cuda::split(gpuPool.d_flow[k], planes, gpuPool.cvStreams[k]);

		writeFlowKernel << <grid, block, 0, gpuPool.streams[k] >> > (
			gpuPool.d_u[k].ptr<float>(),
			gpuPool.d_v[k].ptr<float>(),
			gpuPool.d_U[0],
			gpuPool.d_V[0],
			geo.maxnCamV,
			geo.maxnCamU,
			k);

		cudaEventRecord(gpuPool.events[k], gpuPool.streams[k]);
	}

	// 等待所有 camera 完成
	for (int k = 0; k < geo.numCam; ++k)
		cudaEventSynchronize(gpuPool.events[k]);

	cudaEventRecord(ev_img_end, 0);
	cudaEventSynchronize(ev_img_end);

	float t_img = 0.0f;
	cudaEventElapsedTime(&t_img, ev_img_start, ev_img_end);

	mexPrintf("[GPU] img -> displacement time = %.3f ms\n", t_img);

	// =====================================================
	// ================== reconstruction ===================
	// =====================================================
	
	for (int k = 0; k < geo.numCam; ++k)
		cudaStreamWaitEvent(gpuPool.reconstructionStream, gpuPool.events[k], 0);

	cudaEventRecord(ev_rec_start, gpuPool.reconstructionStream);

	// ================================ uv to eps
	int threads_cam = 128;
	int blocks_cam =
		(geo.numCam + threads_cam - 1) / threads_cam;

	precompute_KRinv_kernel << <blocks_cam, threads_cam >> > (
		gpuPool.d_IMCam,
		gpuPool.d_RrCam,
		gpuPool.d_Zbc,
		gpuPool.d_Zpc,
		gpuPool.d_KR_inv,
		gpuPool.d_scale,
		geo.numCam);

	cudaCheckErrors("precompute_KRinv_kernel failed");

	dim3 block_uv(16, 16);
	dim3 grid_uv(
		(geo.maxnCamU + block_uv.x - 1) / block_uv.x,
		(geo.maxnCamV + block_uv.y - 1) / block_uv.y,
		geo.numCam);


	uvtoeps_kernel_opt << <grid_uv, block_uv, 0, gpuPool.reconstructionStream >> > (
		gpuPool.d_U[0],
		gpuPool.d_V[0],
		gpuPool.d_projectionsob,
		gpuPool.d_KR_inv,
		gpuPool.d_scale,
		geo.numCam,
		geo.maxnCamU,
		geo.maxnCamV);
	cudaCheckErrors("uvtoeps_kernel_opt failed");

	// ================================  IRN-TV-CGLS
	// for build weight
	cudaMemcpy3DParms copyParams = { 0 };
	cudaMemcpy3DParms copyParams_p = { 0 };

	// Algorithm stopped declare
	int remember = 0;
	// for tic toc

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
	int gridSize_4Dflow = (3 * gpuPool.voxelSize + blockSize_1D - 1) / blockSize_1D;
	int gridSize_4Dproj = (3 * gpuPool.projSize + blockSize_1D - 1) / blockSize_1D;
	int gridSize_3Dflow = (gpuPool.voxelSize + blockSize_1D - 1) / blockSize_1D;
	int gridSize_3Dproj = (gpuPool.projSize + blockSize_1D - 1) / blockSize_1D;

	int iter = 0;
	bool early_stop = false;

	while (iter < reconP.niter) {
		copyParams.srcPtr = make_cudaPitchedPtr(gpuPool.d_x, geo.nVoxelX * sizeof(float), geo.nVoxelX, geo.nVoxelY);
		copyParams.dstArray = gpuPool.d_x_array;
		copyParams.extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
		copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&copyParams);
		cudaCheckErrors("tex_x cudaMemcpy3D failed");

		// Call the CUDA kernel to build weights
		build_weights_kernel << <gridSize_flow, blockSize, 0, gpuPool.reconstructionStream >> > (gpuPool.d_W, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, gpuPool.tex_x);
		cudaStreamSynchronize(gpuPool.reconstructionStream);


		cudaCheckErrors("build_weights_kernel fail");

		// Call Axfun kernel
		cudaMemset(gpuPool.d_prox_aux_1, 0, 3 * gpuPool.projSize * sizeof(float));

		Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_prox_aux_1, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
		for (int i = 0; i < 3; ++i) {
			cudaStreamSynchronize(gpuPool.stream_diff[i]);
		}
		cudaCheckErrors("Axfun_kernel fail");

		// Call Lx kernel
		float lambda_sqrt = sqrt(reconP.lambda);
		Lx_kernel << <gridSize_flow, blockSize, 0, gpuPool.reconstructionStream >> > (gpuPool.d_prox_aux_2, gpuPool.d_W, gpuPool.tex_x, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
		cudaCheckErrors("Lx_kernel fail");

		// Call sub kernel
		sub_kernel << <gridSize_4Dproj, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_r_aux_1, gpuPool.d_projectionsob, gpuPool.d_prox_aux_1, 3 * gpuPool.projSize);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
		cudaCheckErrors("sub_kernel failed");

		// Call negate4D kernel
		negate4D_kernel << <gridSize_4Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_r_aux_2, gpuPool.d_prox_aux_2, 3 * gpuPool.voxelSize);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
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
			Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 1);
			Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 2);
			Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 3);
		}
		for (int i = 0; i < 3; ++i)
			cudaStreamSynchronize(gpuPool.stream_diff[i]);

		matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 1);
		matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 2);
		matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_p_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 3);
		for (int i = 0; i < 3; ++i)
			cudaStreamSynchronize(gpuPool.stream_diff[i]);
		cudaCheckErrors("Atbfun_kernel failed");

		// Call Ltx kernel
		Ltx_kernel << <gridSize_flow, blockSize, 0, gpuPool.reconstructionStream >> > (gpuPool.d_p_aux_2, gpuPool.d_W, gpuPool.d_r_aux_2, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
		cudaCheckErrors("Ltx_kernel failed");

		// Call add kernel
		add_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_p, gpuPool.d_p_aux_1, gpuPool.d_p_aux_2, gpuPool.voxelSize);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
		cudaCheckErrors("add_kernel failed");

		// Call norm2 kernel
		cudaMemset(gpuPool.d_gamma, 0, sizeof(float));
		norm2square_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_gamma, gpuPool.d_p, gpuPool.voxelSize);
		cudaStreamSynchronize(gpuPool.reconstructionStream);
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

			cudaMemset(gpuPool.d_q_aux_1, 0, 3 * gpuPool.projSize * sizeof(float));

			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_q_aux_1, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			for (int i = 0; i < 3; ++i) {
				cudaStreamSynchronize(gpuPool.stream_diff[i]);
			}
			cudaCheckErrors("while Axfun_kernel fail");

			// Call Lx kernel
			Lx_kernel << <gridSize_flow, blockSize, 0, gpuPool.reconstructionStream >> > (gpuPool.d_q_aux_2, gpuPool.d_W, gpuPool.tex_x, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while Lx_kernel fail");

			// Call alpha kernel
			cudaMemset(gpuPool.d_gamma_q_aux_1, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dproj, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_gamma_q_aux_1, gpuPool.d_q_aux_1, 3 * gpuPool.projSize);
			cudaMemset(gpuPool.d_gamma_q_aux_2, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_gamma_q_aux_2, gpuPool.d_q_aux_2, 3 * gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			calcAlpha_kernel << <1, 1 >> > (gpuPool.d_alpha, gpuPool.d_gamma, gpuPool.d_gamma_q_aux_1, gpuPool.d_gamma_q_aux_2);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while alpha_kernel failed");

			float alpha_host = 0.0f;
			cudaMemcpy(&alpha_host,
				gpuPool.d_alpha,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: alpha = %e\n", iter-1, alpha_host);

			// Call updateX kernel
			updateX_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_x, gpuPool.d_p, gpuPool.d_alpha, gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while updateX_kernel failed");

			// Call calcAux=sub + Ax kernel
			copyParams.srcPtr = make_cudaPitchedPtr(gpuPool.d_x, geo.nVoxelX * sizeof(float), geo.nVoxelX, geo.nVoxelY);
			copyParams.dstArray = gpuPool.d_x_array;
			copyParams.extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3D(&copyParams);
			cudaCheckErrors("tex_x cudaMemcpy3D failed");

			cudaMemset(gpuPool.d_aux_m, 0, 3 * gpuPool.projSize * sizeof(float));

			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 1, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 2, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			Axfun_kernel << <gridSize_proj, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_aux_m, gpuPool.geoArray[0], 3, gpuPool.tex_x, gpuPool.projParamsArrayDev);
			for (int i = 0; i < 3; ++i) {
				cudaStreamSynchronize(gpuPool.stream_diff[i]);
			}
			cudaCheckErrors("while calcuAux-Axfun_kernel fail");

			sub_kernel << <gridSize_4Dproj, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_aux, gpuPool.d_projectionsob, gpuPool.d_aux_m,  3 * gpuPool.projSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while calcuAux-sub_kernel failed");

			//  Calc resL2
			cudaMemset(gpuPool.d_aux_gamma, 0, sizeof(float));
			norm2square_kernel << <gridSize_4Dproj, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_aux_gamma, gpuPool.d_aux, 3 * gpuPool.projSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			sqrt_kernel << <1, 1, 0, gpuPool.reconstructionStream >> > (gpuPool.d_aux_gamma, gpuPool.d_aux_gamma);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			update_resL2_kernel << <1, 1, 0, gpuPool.reconstructionStream >> > (gpuPool.d_resL2, gpuPool.d_aux_gamma, iter);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while  resL2_kernel failed");

			float resL2_host = 0.0f;
			cudaMemcpy(&resL2_host,
				gpuPool.d_resL2 + iter-1,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: resL2 = %e\n", iter-1, resL2_host);

			// Call check_condition kernel
			check_condition_kernel << <1, 1, 0, gpuPool.reconstructionStream >> > (gpuPool.d_condition, gpuPool.d_resL2, reconP.niter_break, iter);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while  check_condition_kernel failed");
			cudaMemcpy(gpuPool.h_condition, gpuPool.d_condition, sizeof(bool), cudaMemcpyDeviceToHost);
			//// if condition is true
			if (*gpuPool.h_condition)
			{
				// Call updateXsub kernel
				updateXsub_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_x, gpuPool.d_p, gpuPool.d_alpha, gpuPool.voxelSize);
				cudaStreamSynchronize(gpuPool.reconstructionStream);
				cudaCheckErrors("while updateXsub_kernel failed");
				// algorithm stoped declare
				if (remember == iter) {
					cudaMemcpy(result, gpuPool.d_x, gpuPool.voxelSize * sizeof(float), cudaMemcpyDeviceToHost);
					early_stop = true;
					break;
				}
				remember = iter;
				iter--;
				mexPrintf("Orthogonality lost, restarting at iteration %d. \n", iter);
				break;
			}
			// if If step is adecuate, then continue withg CGLS
			// r_aux_1
			updateXsub_kernel << <gridSize_4Dproj, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_r_aux_1, gpuPool.d_q_aux_1, gpuPool.d_alpha, 3 * gpuPool.projSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while update r_aux_1 --> updateXsub_kernel failed");
			// r_aux_2
			updateXsub_kernel << <gridSize_4Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_r_aux_2, gpuPool.d_q_aux_2, gpuPool.d_alpha,  3 * gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
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
				Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 1);
				Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 2);
				Atbfun_kernel << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_image, gpuPool.geoArray[0], proj_j, gpuPool.tex_p, gpuPool.projParamsArray2Dev, gpuPool.projCoeffArray2Dev, 3);
				for (int i = 0; i < 3; ++i)
					cudaStreamSynchronize(gpuPool.stream_diff[i]);
			}


			
			matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[0] >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 1);
			matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[1] >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 2);
			matrixDiffTMultiply << <gridSize_flow, blockSize, 0, gpuPool.stream_diff[2] >> > (gpuPool.d_s_aux_1, gpuPool.geoArray[0], gpuPool.d_image, 3);
			for (int i = 0; i < 3; ++i)
				cudaStreamSynchronize(gpuPool.stream_diff[i]);
			cudaCheckErrors("while calc s_aux_1-- > Atbfun_kernel failed");

			
			// s_aux_2
			cudaMemsetAsync(gpuPool.d_s_aux_2, 0, sizeof(float), gpuPool.reconstructionStream);
			Ltx_kernel << <gridSize_flow, blockSize, 0, gpuPool.reconstructionStream >> > (gpuPool.d_s_aux_2, gpuPool.d_W, gpuPool.d_r_aux_2, geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ, lambda_sqrt);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while calc s_aux_2-- >Ltx_kernel failed");

			
			// s
			add_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_s, gpuPool.d_s_aux_1, gpuPool.d_s_aux_2, gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while calc s-- >add_kernel failed");

			/////////////////////////////////////////////////////////////////
			cudaMemcpyAsync(
				gpuPool.h_r_aux_2,
				gpuPool.d_r_aux_2,
				3*gpuPool.voxelSize * sizeof(float),
				cudaMemcpyDeviceToHost,
				gpuPool.reconstructionStream);

			cudaMemcpyAsync(
				gpuPool.h_W,
				gpuPool.d_W,
				gpuPool.voxelSize * sizeof(float),
				cudaMemcpyDeviceToHost,
				gpuPool.reconstructionStream);

			cudaStreamSynchronize(gpuPool.reconstructionStream);

			char filename[256];
			sprintf(filename, "recon_iter_%04d.mat", iter);
			MATFile* pmat = matOpen(filename, "w");
			if (!pmat) {
				printf("Error creating MAT file %s\n", filename);
				return;
			}

			mwSize dims[3];
			dims[0] = geo.nVoxelX;  // MATLAB: 行
			dims[1] = geo.nVoxelY;  // MATLAB: 列
			dims[2] = 3*geo.nVoxelZ;  // 第三维

			mxArray* mxX = mxCreateNumericArray(
				3, dims, mxSINGLE_CLASS, mxREAL);

			float* mxData = (float*)mxGetData(mxX);
			memcpy(mxData, gpuPool.h_r_aux_2,
				3*gpuPool.voxelSize * sizeof(float));
			matPutVariable(pmat, "rnormal", mxX);
			mxDestroyArray(mxX);

			mxArray* mxW = mxCreateNumericArray(
				3, dims, mxSINGLE_CLASS, mxREAL);

			memcpy(mxGetData(mxW),
				gpuPool.h_W,
				gpuPool.voxelSize * sizeof(float));

			matPutVariable(pmat, "Wnormal", mxW);
			mxDestroyArray(mxW);

			matClose(pmat);
			///////////////////////////////////////////////////////////

			// gamma1
			cudaMemset(gpuPool.d_gamma1, 0, sizeof(float));
			norm2square_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_gamma1, gpuPool.d_s, gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while calc gamma1-- >norm2square_kernel failed");

			float gamma1_host = 0.0f;
			cudaMemcpy(&gamma1_host,
				gpuPool.d_gamma1,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: gamma1 = %e\n", iter, gamma1_host);

			
			// beta
			div_kernel << <1, 1, 0, gpuPool.reconstructionStream >> > (gpuPool.d_beta, gpuPool.d_gamma1, gpuPool.d_gamma);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while calc beta-- >div_kernel failed");

			float oldgamma_host = 0.0f;
			cudaMemcpy(&oldgamma_host,
				gpuPool.d_gamma,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: oldgamma = %e\n", iter - 1, oldgamma_host);

			float beta_host = 0.0f;
			cudaMemcpy(&beta_host,
				gpuPool.d_beta,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: beta = %e\n", iter, beta_host);


			// update gamma
			equal_kernel << <1, 1, 0, gpuPool.reconstructionStream >> > (gpuPool.d_gamma, gpuPool.d_gamma1);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while update gamma-- >equal_kernel failed");

			float newgamma_host = 0.0f;
			cudaMemcpy(&newgamma_host,
				gpuPool.d_gamma,
				sizeof(float),
				cudaMemcpyDeviceToHost);

			printf("iter %d: newgamma = %e\n", iter, newgamma_host);
			// update p
			updateP_kernel << <gridSize_3Dflow, blockSize_1D, 0, gpuPool.reconstructionStream >> > (gpuPool.d_p, gpuPool.d_s, gpuPool.d_beta, gpuPool.voxelSize);
			cudaStreamSynchronize(gpuPool.reconstructionStream);
			cudaCheckErrors("while update p-- >updateP_kernel failed");

			//
			if (reconP.niter_break != 0 && iter % reconP.niter_break == 0) {
				break;
			}
		}
		if (early_stop) break;
	}

	// =====================================================
	// ================== output ===================
	// =====================================================

	/*cudaMemcpyAsync(
		result,
		d_projectionsob,
		3 * geo.numCam * geo.maxnCamU * geo.maxnCamV * sizeof(float),
		cudaMemcpyDeviceToHost,
		reconstructionStream);*/

	cudaMemcpyAsync(
		result,
		gpuPool.d_x,
		gpuPool.voxelSize * sizeof(float),
		cudaMemcpyDeviceToHost,
		gpuPool.reconstructionStream);

	cudaStreamSynchronize(gpuPool.reconstructionStream);


	cudaEventRecord(ev_rec_end, gpuPool.reconstructionStream);
	cudaEventSynchronize(ev_rec_end);

	float t_rec = 0.0f;
	cudaEventElapsedTime(&t_rec, ev_rec_start, ev_rec_end);

	mexPrintf("[GPU] reconstruction (uv -> eps) time = %.3f ms\n", t_rec);
	mexPrintf("[GPU] total GPU time = %.3f ms\n", t_img + t_rec);
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
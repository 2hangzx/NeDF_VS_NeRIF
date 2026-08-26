/*-------------------------------------------------------------------------
 *
 * CUDA functions for backrpojection using match weights
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
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math.h>
#include <device_launch_parameters.h>

#include "BOSLAB_common.h"
#include "voxel_backprojection.h"

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
				mexPrintf("Error: %s\n", msg);\
				mexPrintf("Atb:match: %s\n", cudaGetErrorString(__err)); \
        } \
} while (0)

#define MAXTREADS 1024

 // this definitionmust go here.
void CreateTexture(const GpuIds& gpuids, float* projectiondata, Geometry geo, cudaArray** d_cuArrTex, unsigned int nangles, cudaTextureObject_t *texImage, cudaStream_t* stream, int nStreamDevice, bool allocate);

const int PROJ_PER_KERNEL = 32;  // Number of 2D projections to be analyzed by a single thread. This can be tweaked to see what works best. 32 was the optimal value in the paper by Zinsser and Keck.
const int VOXELS_PER_THREAD = 8;  // Number of voxels to be computed by s single thread. Can be tweaked to see what works best. 4 was the optimal value in the paper by Zinsser and Keck.

// We have PROJ_PER_KERNEL projections and we need 4 parameters for each projection:
//   deltaX, deltaY, deltaZ, xyzOrigin
// So we need to keep PROJ_PER_KERNEL*4 values in our deltas array FOR EACH CALL to our main kernel
// (they will be updated in the main loop before each kernel call).

__constant__ Point3D projParamsArray2Dev[6 * PROJ_PER_KERNEL];  // Dev means it is on device
__constant__ float projCoeffArray2Dev[2 * PROJ_PER_KERNEL];
//______________________________________________________________________________
//
//      Function:       kernelPixelBackprojectionFDK
//
//      Description:    Main FDK backprojection kernel
//______________________________________________________________________________

__global__ void kernelPixelMaskBackprojection(const Geometry geo, float* image, int* d_nCamU,
	int* d_nCamV, const int currProjSetNumber, const int totalNoOfProjections, cudaTextureObject_t tex)
{

	unsigned long long indX = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long long indZ = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned long startIndZ = blockIdx.z * blockDim.z + threadIdx.z;  // This is only STARTING z index of the column of voxels that the thread will handle
	unsigned long long startIndY = blockIdx.z * VOXELS_PER_THREAD + threadIdx.z;  // This is only STARTING z index of the column of voxels that the thread will handle
	//Make sure we don't go out of bounds
	if (indZ >= geo.nVoxelX || indX >= geo.nVoxelX || startIndY >= geo.nVoxelY)
		return;

	// We'll keep a local auxiliary array of values of a column of voxels that this thread will update
	float voxelColumn[VOXELS_PER_THREAD];

	// First we need to copy the curent 3D volume values from the column to our auxiliary array so that we can then
	// work on them (update them by computing values from multiple projections) locally - avoiding main memory reads/writes

	unsigned long colIdx;
#pragma unroll
	for (colIdx = 0; colIdx < VOXELS_PER_THREAD; colIdx++)
	{
		unsigned long long indY = startIndY + colIdx;
		// If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
		// be trying to copy the out of bounds values back to the 3D volume anyway (bounds checks will be done in the final loop where the updated values go back to the main volume)
		if (indY >= geo.nVoxelY)
			break;   // break the loop.

		unsigned long long idx = indY * (unsigned long long)geo.nVoxelZ*(unsigned long long)geo.nVoxelX + indX * (unsigned long long)geo.nVoxelZ + indZ;
		voxelColumn[colIdx] = image[idx];   // Read the current volume value that we'll update by computing values from MULTIPLE projections (not just one)
		// We'll be updating the local (register) variable, avoiding reads/writes from the slow main memory.
	}  // END copy 3D volume voxels to local array

	// Now iterate through projections
#pragma unroll
	for (unsigned long projNumber = 0; projNumber < PROJ_PER_KERNEL; projNumber++)
	{
		// Get the current parameters from parameter arrays in constant memory.
		unsigned long indAlpha = currProjSetNumber * PROJ_PER_KERNEL + projNumber;  // This is the ABSOLUTE projection number in the projection array

		// Our currImageVal will be updated by hovewer many projections we had left in the "remainder" - that's OK.
		if (indAlpha >= totalNoOfProjections)
			break;

		Point3D xyzOrigin = projParamsArray2Dev[6 * projNumber];  // 6*projNumber because we have 6 Point3D values per projection
		Point3D S = projParamsArray2Dev[6 * projNumber + 1];
		Point3D midPtBK = projParamsArray2Dev[6 * projNumber + 2];
		Point3D midDirBK = projParamsArray2Dev[6 * projNumber + 3];
		Point3D deltaU = projParamsArray2Dev[6 * projNumber + 4];
		Point3D deltaV = projParamsArray2Dev[6 * projNumber + 5];


		float DSD = projCoeffArray2Dev[2 * projNumber];
		float co = projCoeffArray2Dev[2 * projNumber + 1];
		// Now iterate through Z in our voxel column FOR A GIVEN PROJECTION
#pragma unroll
		for (colIdx = 0; colIdx < VOXELS_PER_THREAD; colIdx++)
		{
			unsigned long long indY = startIndY + colIdx;

			// If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
			// be trying to copy the out of bounds values anyway (bounds checks will be done in the final loop where the values go to the main volume)
			if (indY >= geo.nVoxelY)
				break;   // break the loop.

			// "XYZ" in the scaled coordinate system of the current point. The image is rotated with the projection angles.
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
				break;
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
				u = (pointBK.x - midPtBK.x) / deltaU.x + d_nCamU[indAlpha] * 0.5f;
			}
			else if (deltaU.z != 0) {
				u = (pointBK.z - midPtBK.z) / deltaU.z + d_nCamU[indAlpha] * 0.5f;
			}
			else if (deltaU.y != 0) {
				u = (pointBK.y - midPtBK.y) / deltaU.y + d_nCamU[indAlpha] * 0.5f;
			}

			if (deltaV.y != 0) {
				v = (pointBK.y - midPtBK.y) / deltaV.y + d_nCamV[indAlpha] * 0.5f;
			}
			else if (deltaU.x != 0) {
				v = (pointBK.x - midPtBK.x) / deltaV.x + d_nCamV[indAlpha] * 0.5f;
			}
			else if (deltaU.z != 0) {
				v = (pointBK.z - midPtBK.z) / deltaV.z + d_nCamV[indAlpha] * 0.5f;
			}

			if (u >= d_nCamU[indAlpha] || v >= d_nCamV[indAlpha]) break;

			float sample = tex3D<float>(tex, v, u, indAlpha + 0.5f);

			float weight = 0;
			//
			//
			//
			// IMPORTANT: The weights are almost 50% of the computational time. Is there a way of speeding this up??
			//
			//Real coordinates of Voxel. Instead of reverting the tranformation, its less math (faster) to compute it from the indexes.
			float L, lsq;

			L = __fsqrt_rd((S.x - pointBK.x)*(S.x - pointBK.x) + (S.y - pointBK.y)*(S.y - pointBK.y) + (S.z - pointBK.z)*(S.z - pointBK.z)); // Sz=0 always.
			lsq = (S.x - P.x)*(S.x - P.x)
				+ (S.y - P.y)*(S.y - P.y)
				+ (S.z - P.z)*(S.z - P.z);
			weight = __fdividef(L*L*L, (DSD*lsq));

			//dx*dy*dz/du/dv


						// Get Value in the computed (U,V) and multiply by the corresponding weight.
						// indAlpha is the ABSOLUTE number of projection in the projection array (NOT the current number of projection set!)

			voxelColumn[colIdx] += sample;

		}  // END iterating through column of voxels

	}  // END iterating through multiple projections

	// And finally copy the updated local voxelColumn array back to our 3D volume (main memory)
#pragma unroll
	for (colIdx = 0; colIdx < VOXELS_PER_THREAD; colIdx++)
	{
		unsigned long long indY = startIndY + colIdx;
		// If we are out of bounds, break the loop. The voxelColumn array will be updated partially, but it is OK, because we won't
		// be trying to copy the out of bounds values back to the 3D volume anyway (bounds checks will be done in the final loop where the values go to the main volume)
		if (indY >= geo.nVoxelY)
			break;   // break the loop.

		unsigned long long idx = indY * (unsigned long long)geo.nVoxelZ*(unsigned long long)geo.nVoxelX + indX * (unsigned long long)geo.nVoxelZ + indZ;
		image[idx] = voxelColumn[colIdx];   // Read the current volume value that we'll update by computing values from MULTIPLE projections (not just one)
		// We'll be updating the local (register) variable, avoiding reads/writes from the slow main memory.
		// According to references (Papenhausen), doing = is better than +=, since += requires main memory read followed by a write.
		// We did all the reads into the local array at the BEGINNING of this kernel. According to Papenhausen, this type of read-write split is
		// better for avoiding memory congestion.
	}  // END copy updated voxels from local array to our 3D volume

}  // END kernelPixelBackprojectionFDK

//______________________________________________________________________________
//
//      Function:       voxel_backprojection
//
//      Description:    Main host function for FDK backprojection (invokes the kernel)
//______________________________________________________________________________

int mask_backprojection(float * projections, Geometry geo, const int diffselect, float* result, const GpuIds& gpuids) {
	// Prepare for MultiGPU
	int deviceCount = gpuids.GetLength();
	cudaCheckErrors("Device query fail");
	if (deviceCount == 0) {
		mexPrintf("Atb:Voxel_backprojection:GPUsemexPrintflect", "There are no available device(s) that support CUDA\n");
	}


	// CODE assumes
	// 1.-All available devices are usable by this code
	// 2.-All available devices are equal, they are the same machine (warning thrown)
	// Check the available devices, and if they are the same
	if (!gpuids.AreEqualDevices()) {
		mexPrintf("Atb:Voxel_backprojection2:GPUselect", "Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed.");
	}

	int dev;

	// Split the CT problem
	unsigned int split_image;
	unsigned int split_projections;
	splitCTbackprojection(gpuids, geo, &split_image, &split_projections);


	// Create the arrays for the geometry. The main difference is that geo.offZ has been tuned for the
	// image slices. The rest of the Geometry is the same
	Geometry* geoArray = (Geometry*)malloc(split_image*deviceCount * sizeof(Geometry));
	createGeoArray(split_image*deviceCount, geo, geoArray);

	// Now lest allocate all the image memory on the GPU, so we can use it later. If we have made our numbers correctly
	// in the previous section this should leave enough space for the textures.
	size_t num_bytes_img = (size_t)geo.nVoxelZ*(size_t)geo.nVoxelX*(size_t)geoArray[0].nVoxelY * sizeof(float);
	float** dimage = (float**)malloc(deviceCount * sizeof(float*));
	float** dimagediff = (float**)malloc(deviceCount * sizeof(float*));
	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaMalloc((void**)&dimage[dev], num_bytes_img);
		cudaMalloc((void**)&dimagediff[dev], num_bytes_img);
		cudaCheckErrors("cudaMalloc fail");
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
	if (isHostRegisterSupported & split_image > 1) {
		cudaHostRegister(result, (size_t)geo.nVoxelZ*(size_t)geo.nVoxelX*(size_t)geo.nVoxelY*(size_t)sizeof(float), cudaHostRegisterPortable);
	}
	if (isHostRegisterSupported) {
		cudaHostRegister(projections, (size_t)geo.maxnCamU*(size_t)geo.maxnCamV*(size_t)geo.numCam*(size_t)sizeof(float), cudaHostRegisterPortable);
	}
	cudaCheckErrors("Error pinning memory");

	// allocate geo.nCamU and geo.nCamV into device
	int* d_nCamU;
	int* d_nCamV;
	cudaMalloc((void**)&d_nCamU, geo.numCam * sizeof(int));
	cudaMalloc((void**)&d_nCamV, geo.numCam * sizeof(int));

	cudaMemcpy(d_nCamU, geo.nCamU, geo.numCam * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nCamV, geo.nCamV, geo.numCam * sizeof(int), cudaMemcpyHostToDevice);

	//If it is the first time, lets make sure our image is zeroed.
	int nStreamDevice = 2;
	int nStreams = deviceCount * nStreamDevice;
	cudaStream_t* stream = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));;

	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		for (int i = 0; i < nStreamDevice; ++i) {
			cudaStreamCreate(&stream[i + dev * nStreamDevice]);

		}
	}

	// Kernel auxiliary variables
	Point3D* projParamsArray2Host;
	cudaMallocHost((void**)&projParamsArray2Host, 6 * PROJ_PER_KERNEL * sizeof(Point3D));
	float* projCoeffArray2Host;
	cudaMallocHost((void**)&projCoeffArray2Host, 2 * PROJ_PER_KERNEL * sizeof(float));


	// Texture object variables
	cudaTextureObject_t *texProj;
	cudaArray **d_cuArrTex;
	texProj = (cudaTextureObject_t*)malloc(deviceCount * 2 * sizeof(cudaTextureObject_t));
	d_cuArrTex = (cudaArray**)malloc(deviceCount * 2 * sizeof(cudaArray*));



	unsigned int proj_split_overlap_number;
	// Start with the main loop. The Projection data needs to be allocated and dealocated in the main loop
	// as due to the nature of cudaArrays, we can not reuse them. This should not be a problem for the fast execution
	// of the code, as repeated allocation and deallocation only happens when the projection data is very very big,
	// and therefore allcoation time should be negligible, fluctuation of other computations should mask the time.
	unsigned long long proj_linear_idx_start;
	unsigned int current_proj_split_size, current_proj_overlap_split_size;
	size_t num_bytes_img_curr;
	size_t img_linear_idx_start;
	float** partial_projection;
	size_t* proj_split_size;

	for (unsigned int img_slice = 0; img_slice < split_image; img_slice++) {
		//
				// Initialize the memory if its the first time.
		for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			cudaMemset(dimage[dev], 0, num_bytes_img);
			cudaMemset(dimagediff[dev], 0, num_bytes_img);
			cudaCheckErrors("memset fail");
		}

		for (unsigned int proj = 0; proj < split_projections; proj++) {


			// What is the size of the current chunk of proejctions we need in?
			current_proj_split_size = (geo.numCam + split_projections - 1) / split_projections;
			// if its the last one its probably less
			current_proj_split_size = ((proj + 1)*current_proj_split_size < geo.numCam) ? current_proj_split_size : geo.numCam - current_proj_split_size * proj;

			// We are going to split it in the same amount of kernels we need to execute.
			proj_split_overlap_number = (current_proj_split_size + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;

			// Create pointer to pointers of projections and precompute their location and size.
			if (!proj && !img_slice) {
				partial_projection = (float**)malloc(current_proj_split_size * sizeof(float*));
				proj_split_size = (size_t*)malloc(current_proj_split_size * sizeof(size_t*));
			}
			for (unsigned int proj_block_split = 0; proj_block_split < proj_split_overlap_number; proj_block_split++) {
				// Crop the last one, as its likely its not completely divisible.
				// now lets split this for simultanoeus memcopy and compute.
				// We want to make sure that if we can, we run PROJ_PER_KERNEL projections, to maximize kernel acceleration
				// current_proj_overlap_split_size units = angles
				current_proj_overlap_split_size = max((current_proj_split_size + proj_split_overlap_number - 1) / proj_split_overlap_number, PROJ_PER_KERNEL);
				current_proj_overlap_split_size = (proj_block_split < proj_split_overlap_number - 1) ? current_proj_overlap_split_size : current_proj_split_size - (proj_split_overlap_number - 1)*current_proj_overlap_split_size;
				//Get the linear index where the current memory chunk starts.

				proj_linear_idx_start = (unsigned long long)((geo.numCam + split_projections - 1) / split_projections)*(unsigned long long)proj*(unsigned long long)geo.maxnCamU*(unsigned long long)geo.maxnCamV;
				proj_linear_idx_start += proj_block_split * max((current_proj_split_size + proj_split_overlap_number - 1) / proj_split_overlap_number, PROJ_PER_KERNEL)*(unsigned long long)geo.maxnCamU*(unsigned long long)geo.maxnCamV;
				//Store result
				proj_split_size[proj_block_split] = current_proj_overlap_split_size;
				partial_projection[proj_block_split] = &projections[proj_linear_idx_start];

			}


			for (unsigned int proj_block_split = 0; proj_block_split < proj_split_overlap_number; proj_block_split++) {


				// Now get the projections on memory

				CreateTexture(gpuids,
					partial_projection[proj_block_split], geo,
					&d_cuArrTex[(proj_block_split % 2)*deviceCount],
					proj_split_size[proj_block_split],
					&texProj[(proj_block_split % 2)*deviceCount],
					stream, nStreamDevice,
					(proj_block_split < 2) & !proj & !img_slice);// Only allocate if its the first 2 calls

				for (dev = 0; dev < deviceCount; dev++) {
					cudaSetDevice(gpuids[dev]);
					cudaStreamSynchronize(stream[dev*nStreamDevice + 1]);
				}

				for (dev = 0; dev < deviceCount; dev++) {
					//Safety:
					// Depends on the amount of GPUs, the case where a image slice is zero hight can happen.
					// Just break the loop if we reached that point
					if (geoArray[img_slice*deviceCount + dev].nVoxelY == 0)
						break;

					cudaSetDevice(gpuids[dev]);



					int divx, divy, divz;
					// RB: Use the optimal (in their tests) block size from paper by Zinsser and Keck (16 in x and 32 in y).
					// I tried different sizes and shapes of blocks (tiles), but it does not appear to significantly affect trhoughput, so
					// let's stick with the values from Zinsser and Keck.
					divx = 16;
					divy = 32;
					divz = VOXELS_PER_THREAD;      // We now only have 32 x 16 threads per block (flat tile, see below), BUT each thread works on a Z column of VOXELS_PER_THREAD voxels, so we effectively need fewer blocks!


					dim3 grid((geo.nVoxelZ + divx - 1) / divx,
						(geo.nVoxelX + divy - 1) / divy,
						(geoArray[img_slice*deviceCount + dev].nVoxelY + divz - 1) / divz);

					dim3 block(divx, divy, 1);    // Note that we have 1 in the Z size, not divz, since each thread works on a vertical set of VOXELS_PER_THREAD voxels (so we only need a "flat" tile of threads, with depth of 1)
					//////////////////////////////////////////////////////////////////////////////////////
					// Main reconstruction loop: go through projections (rotation angles) and backproject
					//////////////////////////////////////////////////////////////////////////////////////

					// Since we'll have multiple projections processed by a SINGLE kernel call, compute how many
					// kernel calls we'll need altogether.
					unsigned int noOfKernelCalls = (proj_split_size[proj_block_split] + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL;  // We'll take care of bounds checking inside the loop if nalpha is not divisible by PROJ_PER_KERNEL
					for (unsigned int i = 0; i < noOfKernelCalls; i++) {

						// Now we need to generate and copy all data for PROJ_PER_KERNEL projections to constant memory so that our kernel can use it
						unsigned int j;
						for (j = 0; j < PROJ_PER_KERNEL; j++) {

							unsigned int currProjNumber_slice = i * PROJ_PER_KERNEL + j;
							unsigned int currProjNumber_global = i * PROJ_PER_KERNEL + j                                                                          // index within kernel
								+ proj * (geo.numCam + split_projections - 1) / split_projections                                          // index of the global projection split
								+ proj_block_split * max(current_proj_split_size / proj_split_overlap_number, PROJ_PER_KERNEL); // indexof overlap current split
							if (currProjNumber_slice >= proj_split_size[proj_block_split])
								break;  // Exit the loop. Even when we leave the param arrays only partially filled, this is OK, since the kernel will check bounds anyway.
							if (currProjNumber_global >= geo.numCam)
								break;  // Exit the loop. Even when we leave the param arrays only partially filled, this is OK, since the kernel will check bounds anyway.

							Point3D xyzOrigin, source, midPtBK, midDirBK, deltaU, deltaV;

							computeDeltasCube(geoArray[img_slice*deviceCount + dev], currProjNumber_global, &xyzOrigin, &source, &midPtBK, &midDirBK, &deltaU, &deltaV);

							//computeInterp(geoArray[img_slice*deviceCount + dev], xyzOrigin, source, midPtBK, midDirBK, deltaU, deltaV);

							projParamsArray2Host[6 * j] = xyzOrigin;		// 7*j because we have 7 Point3D values per projection
							projParamsArray2Host[6 * j + 1] = source;
							projParamsArray2Host[6 * j + 2] = midPtBK;
							projParamsArray2Host[6 * j + 3] = midDirBK;
							projParamsArray2Host[6 * j + 4] = deltaU;
							projParamsArray2Host[6 * j + 5] = deltaV;

							float DSD, co;
							computeBackgroundCoef(geoArray[img_slice*deviceCount + dev], currProjNumber_global, source, midPtBK, &DSD, &co);

							projCoeffArray2Host[2 * j] = DSD;
							projCoeffArray2Host[2 * j + 1] = co;


						}   // END for (preparing params for kernel call)

						// Copy the prepared parameter arrays to constant memory to make it available for the kernel
						cudaMemcpyToSymbolAsync(projParamsArray2Dev, projParamsArray2Host, sizeof(Point3D) * 6 * PROJ_PER_KERNEL, 0, cudaMemcpyHostToDevice, stream[dev*nStreamDevice]);
						cudaMemcpyToSymbolAsync(projCoeffArray2Dev, projCoeffArray2Host, sizeof(float) * 2 * PROJ_PER_KERNEL, 0, cudaMemcpyHostToDevice, stream[dev*nStreamDevice]);
						cudaStreamSynchronize(stream[dev*nStreamDevice]);
						cudaCheckErrors("kernel fail");
						kernelPixelMaskBackprojection << <grid, block, 0, stream[dev*nStreamDevice] >> > (geoArray[img_slice*deviceCount + dev], dimage[dev], d_nCamU, d_nCamV, i, proj_split_size[proj_block_split], texProj[(proj_block_split % 2)*deviceCount + dev]);

					}  // END for
					//////////////////////////////////////////////////////////////////////////////////////
					// END RB code, Main reconstruction loop: go through projections (rotation angles) and backproject
					//////////////////////////////////////////////////////////////////////////////////////
				}
			} // END sub-split of current projection chunk

		} // END projection splits

		// 定义块和网格的大小
		dim3 block(2, 2, 2); // 每个块包含 2x2x2 个线程
		dim3 grid((geo.nVoxelZ + block.x - 1) / block.x,
			(geo.nVoxelX + block.y - 1) / block.y,
			(geo.nVoxelY + block.z - 1) / block.z);

		for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			matrixDiffTMultiply << <grid, block, 0, stream[dev*nStreamDevice] >> > (geoArray[img_slice*deviceCount + dev], dimage[dev], dimagediff[dev], diffselect);
			cudaCheckErrors("kernel fail2");
		}

		// Now we need to take the image out of the GPU
		for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			cudaStreamSynchronize(stream[dev*nStreamDevice]);

			num_bytes_img_curr = (size_t)geoArray[img_slice*deviceCount + dev].nVoxelZ*(size_t)geoArray[img_slice*deviceCount + dev].nVoxelX*(size_t)geoArray[img_slice*deviceCount + dev].nVoxelY * sizeof(float);
			img_linear_idx_start = (size_t)geo.nVoxelZ*(size_t)geo.nVoxelX*(size_t)geoArray[0].nVoxelY*(size_t)(img_slice*deviceCount + dev);
			cudaMemcpyAsync(&result[img_linear_idx_start], dimagediff[dev], num_bytes_img_curr, cudaMemcpyDeviceToHost, stream[dev*nStreamDevice + 1]);
		}
	} // end image splits

	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaDeviceSynchronize();
	}


	// Clean the GPU
	bool two_buffers_used = ((((geo.numCam + split_projections - 1) / split_projections) + PROJ_PER_KERNEL - 1) / PROJ_PER_KERNEL) > 1;
	for (unsigned int i = 0; i < 2; i++) { // 2 buffers (if needed, maybe only 1)
		if (!two_buffers_used && i == 1)
			break;        for (dev = 0; dev < deviceCount; dev++) {
			cudaSetDevice(gpuids[dev]);
			cudaDestroyTextureObject(texProj[i*deviceCount + dev]);
			cudaFreeArray(d_cuArrTex[i*deviceCount + dev]);
		}
	}


	for (dev = 0; dev < deviceCount; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaFree(dimage[dev]);
		cudaFree(dimagediff[dev]);
	}

	cudaFreeHost(projParamsArray2Host);
	free(partial_projection);
	free(proj_split_size);

	free(geoArray);
#ifndef NO_PINNED_MEMORY     
	if (isHostRegisterSupported & split_image > 1) {
		cudaHostUnregister(result);
	}
	if (isHostRegisterSupported) {
		cudaHostUnregister(projections);
	}
#endif 
	for (int i = 0; i < nStreams; ++i)
		cudaStreamDestroy(stream[i]);

	cudaCheckErrors("cudaFree fail");

	//     cudaDeviceReset(); // For the Nvidia Visual Profiler
	return 0;

}  // END voxel_backprojection





void CreateTexture(const GpuIds& gpuids, float* projectiondata, Geometry geo, cudaArray** d_cuArrTex, unsigned int nangles, cudaTextureObject_t *texImage, cudaStream_t* stream, int nStreamDevice, bool allocate) {
	//size_t size_image=geo.nVoxelX*geo.nVoxelY*geo.nVoxelZ;
	int num_devices = gpuids.GetLength();
	const cudaExtent extent = make_cudaExtent(geo.maxnCamV, geo.maxnCamU, nangles);
	if (allocate) {
		for (unsigned int dev = 0; dev < num_devices; dev++) {
			cudaSetDevice(gpuids[dev]);

			//cudaArray Descriptor
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
			//cuda Array
			cudaMalloc3DArray(&d_cuArrTex[dev], &channelDesc, extent);

		}
	}
	for (unsigned int dev = 0; dev < num_devices; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaMemcpy3DParms copyParams = { 0 };
		//Array creation
		copyParams.srcPtr = make_cudaPitchedPtr((void *)projectiondata, extent.width * sizeof(float), extent.width, extent.height);
		copyParams.dstArray = d_cuArrTex[dev];
		copyParams.extent = extent;
		copyParams.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3DAsync(&copyParams, stream[dev*nStreamDevice + 1]);
	}

	//Array creation End
	for (unsigned int dev = 0; dev < num_devices; dev++) {
		cudaSetDevice(gpuids[dev]);
		cudaResourceDesc    texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArrTex[dev];
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = false;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject(&texImage[dev], &texRes, &texDescr, NULL);
	}
}

void splitCTbackprojection(const GpuIds& gpuids, Geometry geo, unsigned int* split_image, unsigned int * split_projections) {


	// We don't know if the devices are being used. lets check that. and only use the amount of memory we need.

	size_t mem_GPU_global;
	checkFreeMemory(gpuids, &mem_GPU_global);

	const int deviceCount = gpuids.GetLength();

	// Compute how much memory each of the relevant memory pieces need
	size_t mem_image = (unsigned long long)geo.nVoxelZ*(unsigned long long)geo.nVoxelX*(unsigned long long)geo.nVoxelY * sizeof(float);
	size_t mem_proj = (unsigned long long)geo.maxnCamU*(unsigned long long)geo.maxnCamV * sizeof(float);




	// Does everything fit in the GPU?

	if (mem_image / deviceCount + mem_proj * PROJ_PER_KERNEL * 2 < mem_GPU_global) {
		// We only need to split if we have extra GPUs
		*split_image = 1;
		*split_projections = 1;
	}
	// We know we need to split, but:
	// Does all the image fit in the GPU, with some slack for a stack of projections??
	else
	{
		// As we can overlap memcpys from H2D of the projections, we should then minimize the amount of image splits.
		// Lets assume to start with that we only need 1 stack of PROJ_PER_KERNEL projections. The rest is for the image.
		size_t mem_free = mem_GPU_global - 2 * mem_proj*PROJ_PER_KERNEL;

		*split_image = (mem_image / deviceCount + mem_free - 1) / mem_free;
		// Now knowing how many splits we have for images, we can recompute how many slices of projections actually
		// fit on the GPU. Must be more than 0 obviously.

		mem_free = mem_GPU_global - (mem_image / deviceCount) / (*split_image); // NOTE: There is some rounding error, but its in the order of bytes, and we have 5% of GPU free jsut in case. We are safe


		*split_projections = (mem_proj*PROJ_PER_KERNEL * 2 + mem_free - 1) / mem_free;

	}
}

// ______________________________________________________________________________
//
//      Function:       createGeoArray
//
//      Description:    This code generates the geometries needed to split the image properly in
//                      cases where the entire image does not fit in the memory of the GPU
//______________________________________________________________________________

void createGeoArray(unsigned int image_splits, Geometry geo, Geometry* geoArray) {


	unsigned int  splitsize = (geo.nVoxelY + image_splits - 1) / image_splits;

	for (unsigned int sp = 0; sp < image_splits; sp++) {
		geoArray[sp] = geo;
		// All of them are splitsize, but the last one, possible
		geoArray[sp].nVoxelY = ((sp + 1)*splitsize < geo.nVoxelY) ? splitsize : max(geo.nVoxelY - splitsize * sp, 0);
		geoArray[sp].sVoxelY = geoArray[sp].nVoxelY* geoArray[sp].dVoxelY;
	}

}

//______________________________________________________________________________
//
//      Function:       computeDeltasCube
//
//      Description:    Computes relative increments for each projection (volume rotation).
//						Increments get passed to the backprojection kernel.
//______________________________________________________________________________

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


void computeInterp(Geometry geo, Point3D xyzOrigin, Point3D S, Point3D midPtBK, Point3D midDirBK, Point3D deltaU, Point3D deltaV)
{
	int indX = 64, indY = 64, indZ = 64;
	int projNumber = 0;
	// "XYZ" in the scaled coordinate system of the current point. The image is rotated with the projection angles.
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
		mexPrintf("fm=0\n");
	}
	else {
		fz = (midPtBK.x - P.x)*midDirBK.x + (midPtBK.y - P.y)*midDirBK.y + (midPtBK.z - P.z)*midDirBK.z;
		t = fz / fm;
	}

	Point3D pointBK;
	pointBK.x = vectX * t + P.x;
	pointBK.y = vectY * t + P.y;
	pointBK.z = vectZ * t + P.z;



	float DSD = sqrt((midPtBK.x - S.x)*(midPtBK.x - S.x) + (midPtBK.y - S.y)*(midPtBK.y - S.y) + (midPtBK.z - S.z)*(midPtBK.z - S.z));


	float u, v;

	if (deltaU.x != 0) {
		u = (pointBK.x - midPtBK.x) / deltaU.x + 1280 * 0.5f;
	}
	else if (deltaU.z != 0) {
		u = (pointBK.z - midPtBK.z) / deltaU.z + 1280 * 0.5f;
	}
	else if (deltaU.y != 0) {
		u = (pointBK.y - midPtBK.y) / deltaU.y + 1280 * 0.5f;
	}

	if (deltaV.y != 0) {
		v = (pointBK.y - midPtBK.y) / deltaV.y + 1024 * 0.5f;
	}
	else if (deltaU.x != 0) {
		v = (pointBK.x - midPtBK.x) / deltaV.x + 1024 * 0.5f;
	}
	else if (deltaU.z != 0) {
		v = (pointBK.z - midPtBK.z) / deltaV.z + 1024 * 0.5f;
	}


	float weight = 0;
	//
	//
	//
	// IMPORTANT: The weights are almost 50% of the computational time. Is there a way of speeding this up??
	//
	//Real coordinates of Voxel. Instead of reverting the tranformation, its less math (faster) to compute it from the indexes.
	float L, lsq;

	L = sqrt((S.x - pointBK.x)*(S.x - pointBK.x) + (S.y - pointBK.y)*(S.y - pointBK.y) + (S.z - pointBK.z)*(S.z - pointBK.z)); // Sz=0 always.
	lsq = (S.x - P.x)*(S.x - P.x)
		+ (S.y - P.y)*(S.y - P.y)
		+ (S.z - P.z)*(S.z - P.z);
	weight = L * L*L / (DSD*lsq);
}



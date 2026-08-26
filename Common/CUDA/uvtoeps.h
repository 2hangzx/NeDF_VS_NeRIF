/*-------------------------------------------------------------------------
 *
 * Header CUDA functions for uvtoeps
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
void uvtoeps(Geometry geo, float* projectionsU, float* projectionsV, float* result, const GpuIds& gpuids, GPUBufferPool& gpuPool);
void matMul3x3(const float* A, const float* B, float* C);
void inv3x3(const float* A, float* invA);
void transpose3x3(const float* A, float* At);
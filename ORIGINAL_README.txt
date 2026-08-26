This work presents a novel neural framework for reconstructing three-dimensional refractive-index fields from background-oriented schlieren measurements of flames.

To facilitate reproducibility, we release the source code of this work, built upon the CT discrete reconstruction implementation of Biguri et al. (TIGRE) [1] and the neural graphics primitives implementation of Tang et al. (iNGP) [2]. Many thanks to their works!
[1].Biguri A, Dosanjh M, Hancock S, et al. TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction[J]. Biomedical Physics & Engineering Express, 2016, 2(5): 055010. https://github.com/CERN/TIGRE
[2].Tang J, Chen X, Wang J, et al. Compressible-composable nerf via rank-residual decomposition[J]. Advances in Neural Information Processing Systems, 2022, 35: 14798-14809. https://github.com/ashawkey/torch-ngp


% First run:
%   1) step1_InitBOSLAB.m
%   2) step2_Compile.m
%
% After successful compilation:
%   Run step1_InitBOSLAB.m, then directly run
%   step3_generate_phantom1_synthetic_data in the demo folder.
%
% Copy the dataset from 'MATLAB/Test_data/Phantom 1' to
% 'PYTHON/NIR-BOS/data'.
%
% Set up the Python environment using environment.yml.
%

# Before running main_bos.py

# Set sys.argv based on the outputs of matlab/step3_generate_phantom1_synthetic_data.m
# Example configuration:
# --scale: 0.00054421
# --ROIsize: [0.95237, 2, 0.95237]
# --ROInum: [140, 294, 140]
# --ROIvoxelsize: 0.013605
# --valbound: [-1, 2.9339]  Due to uncertainties in practical usage, the valbound range can be appropriately relaxed.

# Set the output directory for saving results
# Example: results/phantom1/freencode_disc_mask

# If '--maskflag' is removed from sys.argv, the 3D mask is disabled,
# while the 2D mask-based ray sampling strategy remains enabled.
# To disable the 2D mask-based ray sampling strategy,
# remove UVROIs and masks from the input of the collate function in provider.py:
# rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays,
#                  error_map, self.opt.patch_size)


# Select encoding method in NeRF network initialization
model = NeRFNetwork(
    encoding="Hash", #or Fourier
    bound=opt.bound,
    cuda_ray=opt.cuda_ray,
    density_scale=1,
    min_near=opt.min_near,
    density_thresh=opt.density_thresh,
    mask3Ddata=mask3Ddata,
    ROIsize=opt.ROIsize,
    ROInum=opt.ROInum,
    ROIvoxelsize=opt.ROIvoxelsize,
    valbound=opt.valbound
)

# More detailed encoding parameters can be configured in encoding.py (get_encoder function).
# multires controls the number of frequency encoding levels,
# and log2_hashmap_size defines the hash table size for hash encoding.

# The eval_interval parameter in the Trainer controls the evaluation frequency.
# Every eval_interval steps/epochs, the current predicted refractive index (sigma0)
# and the automatically differentiated refractive index gradients (dsigma_dxyz_auto0)
# are exported as sigmas0_epochMATLAB (.mat) files named as sigmas0_epoch number.


# The gradient computation in the loss function is controlled in utils.py (train_step function).
# It is defined as:
# loss = 1 * loss.mean() + 0 * loss_auto.mean()
# where loss represents the discrete gradient loss,
# and loss_auto represents the gradient loss computed via automatic differentiation.


# On first run of main_BOS.py, CUDA kernels will be compiled (JIT compilation),
# which may take some time.
# This is a normal and expected step.
# If any issues occur and cannot be resolved, please contact us for support.

# After successful compilation, two figures will be displayed:
# the first shows the test set camera and flow field geometry,
# and the second shows the validation set geometry.
# If the result is inconsistent with the expected setup, it is likely due to
# an issue in the MATLAB geometry definition and should be checked.
# Close the figure to proceed to the next step

# After training is completed, an additional figure will be displayed,
# showing the spatial configuration of the camera and flow field for the test set.
# Close the figure to proceed to the next step

# It should be noted that the encoding module is not implemented with CUDA acceleration.
# As a result, hash-based encoding is significantly slower than frequency-based encoding.
# A CUDA-accelerated version of the encoding will be released in the future.


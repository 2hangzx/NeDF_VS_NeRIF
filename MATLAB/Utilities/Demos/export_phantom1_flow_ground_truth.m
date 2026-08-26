function outputPath = export_phantom1_flow_ground_truth(varargin)
%EXPORT_PHANTOM1_FLOW_GROUND_TRUTH Export the exact MATLAB-preprocessed GT.
%
% This is a lightweight interface for Python comparison scripts. It repeats
% only the ground-truth preprocessing used by
% step3_generate_phantom1_synthetic_data.m:
%   1. imresize3(n, 0.5)
%   2. center crop to 140 x 294 x 140
%   3. flow_gt = (n / n0 - 1) / flow_max
%
% It does not run projection generation, CGLS, or neural reconstruction and
% does not modify the original step3 script.
%
% Usage:
%   export_phantom1_flow_ground_truth
%   export_phantom1_flow_ground_truth('OutputPath', 'D:\tmp\flow_gt.mat')

scriptDir = fileparts(mfilename('fullpath'));
matlabRoot = fullfile(scriptDir, '..', '..');
testDataRoot = fullfile(matlabRoot, 'Test_data', 'Phantom 1');
inputPath = fullfile(testDataRoot, 'n_GroundTruth.mat');
defaultOutputPath = fullfile(testDataRoot, '140x294x140', ...
    'flow_ground_truth.mat');

parser = inputParser;
parser.addParameter('OutputPath', defaultOutputPath, ...
    @(value) ischar(value) || isstring(value));
parser.parse(varargin{:});
outputPath = char(parser.Results.OutputPath);

% defaultGeometry.m is needed only to keep the target grid definition in
% one place. No MEX/CUDA initialization is required by this interface.
addpath(fullfile(matlabRoot, 'Utilities'));

geo = defaultGeometry( ...
    'nVoxel', [70; 147; 70] * 2, ...
    'sVoxel', [70; 147; 70], ...
    'angles', linspace(0, 165, 12), ...
    'nCam', [720; 1280], ...
    'dCam', [0.02; 0.02], ...
    'fCam', 105, ...
    'Zbc', 1600, ...
    'Zpc', 800);

if ~isfile(inputPath)
    error('Ground-truth file not found: %s', inputPath);
end
values = load(inputPath, 'n');
if ~isfield(values, 'n')
    error('Variable ''n'' is missing from %s', inputPath);
end

% Keep these operations identical to step3.
nResized = imresize3(values.n, 0.5);
sourceSize = size(nResized);
targetSize = double(geo.nVoxel(:).');
cropTotal = sourceSize - targetSize;
if any(cropTotal < 0) || any(mod(cropTotal, 2) ~= 0)
    error('Cannot center-crop resized GT of size [%s] to [%s].', ...
        num2str(sourceSize), num2str(targetSize));
end
cropStart = cropTotal / 2 + 1;
cropEnd = cropStart + targetSize - 1;
nRoi = nResized( ...
    cropStart(1):cropEnd(1), ...
    cropStart(2):cropEnd(2), ...
    cropStart(3):cropEnd(3));

T0 = 1100;
n0 = 296.15 * (1.00027 - 1) / T0 + 1;
flow0 = nRoi / n0 - 1;
flow_max = abs(min(flow0(:)));
flow_gt = single(flow0 / flow_max);

% Reproduce the NeRF coordinate scaling from step3 so Python can verify the
% comparison space instead of relying on an undocumented hard-coded value.
pixelUnit = geo.dCam(1, 1);
unitVoxelSize = geo.dVoxel / pixelUnit;
unitVolumeSize = geo.sVoxel / pixelUnit;
scale = floor(4 / max(unitVolumeSize) * 1e8) / 1e8;
spacing = double((unitVoxelSize * scale).');
roi_size = double((unitVolumeSize * scale / 2).');
roi_num = targetSize;
resize_factor = 0.5;

outputDir = fileparts(outputPath);
if ~isfolder(outputDir)
    mkdir(outputDir);
end
save(outputPath, 'flow_gt', 'n0', 'flow_max', 'spacing', ...
    'roi_size', 'roi_num', 'scale', 'resize_factor', '-v7');

fprintf('Exported exact flow GT: %s\n', outputPath);
fprintf('shape = [%s], spacing = [%s]\n', ...
    num2str(size(flow_gt)), num2str(spacing, '%.8f '));
end

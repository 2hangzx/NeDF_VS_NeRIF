%--------------------------------------------------------------------------
% Compared with step3_generate_phantom1_synthetic_data.m, which requires a 
% circumferential camera arrangement,this demo provides a transformation 
% from MATLAB camera calibration parameters to BOS geometry configuration.
%
% MATLAB camera calibration parameter acquisition for BOS is not the focus of this project.
% If needed, please contact us for the open-source implementation.
%--------------------------------------------------------------------------
% This file is part of the BOSLAB Toolbox
%
% Copyright (c) 2024, Beihang University
%
% Author:        Wei Hu
% Contact:       weihu22@buaa.edu.cn
%
% This code is intended for academic and research purposes only.
%--------------------------------------------------------------------------
%% Initialize
clear;
close all;
%% read calibaration data
load('.\Test_data\calibration_demo.mat');
cam_num = calibration_result.cam_num;
cam_order = calibration_result.cam_order;
camera_calibration = calibration_result.camera;
background_calibration = calibration_result.background;
%% set geometry
geo.dVoxel = [0.5;0.5;0.5];             % size of each voxel                        (mm)
geo.nVoxel = [56; 64; 56];                 % number of voxels                          (vx)
geo.sVoxel = geo.nVoxel .* geo.dVoxel;     % size of each voxel                        (mm)
geo.Opr = calibration_result.probe_area_center';                          % probe area center position                (mm)
geo.Opr(2) = geo.Opr(2)+10;
geo.Opr(1) = geo.Opr(1)-14;
% camera & background
geo.numCam = cam_num;				% number of cameras             
geo.nCam = repmat([1280;1024],1,geo.numCam);% number of pixels                          (px)     
geo.dCam = repmat([0.0048;0.0048],1,geo.numCam);   
                                            % size of each pixel                        (mm)                                           
geo.sCam = geo.nCam.*geo.dCam;              % total size of the detector                (mm)
geo.fCam = repmat(50,1,geo.numCam);         % focal length                              (mm)
% calculate camera parameter in the individual camera coordinate system
for i = 1: 1: geo.numCam
    geo.IMCam(:, 3*(i-1)+1: 3*i) =camera_calibration(i).background.IntrinsicMatrix';           
                                            % intrinsic matrix
    geo.RCam(:, 3*(i-1)+1: 3*i) = camera_calibration(i).background.RotationMatrices(:, :, 1)';
    geo.TCam(:, i) = camera_calibration(i).background.TranslationVectors(1, :)';

    geo.RrCam(:, 3*(i-1)+1: 3*i) = camera_calibration(i).multi_optimal.Rotation_matrix';
    geo.TrCam(:, i) = camera_calibration(i).multi_optimal.Translation_matrix'+geo.RrCam(:, 3*(i-1)+1: 3*i)*geo.Opr;
end
% fd distance between img plane and cam center (mm) 
for i = 1: 1: geo.numCam
    IMCam = geo.IMCam(:, 3*(i-1)+1: 3*i);
    geo.fd(1,i)= IMCam(1,1)*geo.dCam(1,i);
end

% calculate camera relative parameter in the reference camera coordinate system
for i = 1: 1: geo.numCam    
    geo.Ocr(:,i) = -geo.RrCam(:, 3*(i-1)+1: 3*i)'*geo.TrCam(:, i);
end
geo.Opr = geo.Opr - geo.Opr;
% Zpc distance between probe area center and cam center        (mm)
for i = 1: 1: geo.numCam
    geo.Zpc(:,i) = sqrt( sum( (geo.Ocr(:,i) - geo.Opr).^2 ) );
end
% Zbc distance between bk and cam center        (mm) 
geo.Zbc = 1./ ( 1./geo.fCam - 1./geo.fd );  

% corner position of background
for i = 1: 1: geo.numCam
    uroi = 1;
    vroi = 1;
    widthroi = geo.nCam(1,i);
    heightroi = geo.nCam(2,i);
    v = [vroi,vroi+heightroi-1];
    u = [uroi,uroi+widthroi-1];
    [UROI,VROI] = meshgrid(u,v);
    Zc = repmat(geo.Zbc(i), size(UROI));
    Zc = Zc(:)';
    UROI = UROI(:)';
    VROI = VROI(:)';
    uv1 = [UROI.* Zc; VROI.* Zc; Zc];
    bkIXYZ = inv(geo.IMCam(:,3*(i-1)+1: 3*i)) * uv1;
    bkRXYZ = geo.RrCam(:, 3*(i-1)+1: 3*i)'*(bkIXYZ - geo.TrCam(:,i));
    geo.PbkCorner(:,4*(i-1)+1: 4*i) = bkRXYZ;
end
geo=checkGeo(geo);
plotgeometry(geo);
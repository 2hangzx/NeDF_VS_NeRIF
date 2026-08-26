%--------------------------------------------------------------------------
% BOSLAB Toolbox Initialization
%
% This function initializes the BOSLAB toolbox, including environment setup,
% path configuration, and required dependencies.
%
%
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
%% Add tolbox folders
addpath('./Algorithms');
addpath('./Utilities');
addpath('./Utilities/Quality_measures');
addpath('./Utilities/Geometry');
addpath('./Utilities/Demos');
addpath('./Utilities/GPU');
addpath('./Mex_files/win64');
addpath('./Utilities/Demos');
addpath('./Utilities/Backgrounds');

addpath('./Third_party_tools/Tools');
addpath('./Third_party_tools/sec2hours');



addpath(genpath('./Test_data'));

% different arch versions
if ispc
    if ~isempty(strfind(computer('arch'),'64'))
        addpath('./Mex_files/win64');
    else
        addpath('./Mex_files/win32');
    end
elseif ismac
    if ~isempty(strfind(computer('arch'),'64'))
        addpath('./Mex_files/mac64');
    else
        addpath('./Mex_files/mac32');
    end
else
    if ~isempty(strfind(computer('arch'),'64'))
        addpath('./Mex_files/linux64');
    else
        addpath('./Mex_files/linux32');
    end
end
    
%%
if ispc
    [user, sys]=memory;
    
    if sys.PhysicalMemory.Total<9000000000 % 8Gb
        warning('Your Computer has 8Gb or less of RAM memory. Using image sizes of higher than 512^3 is not recomended (most likely not possible)')
    end
    
    if sys.PhysicalMemory.Total<2500000000 % 2Gb
        warning('Your Computer has 2Gb or less of RAM memory. Using image sizes of higher than 256^3 is not recomended (most likely not possible)')
    end
else
    warning('TIGRE needs a big amount of memory, be careful when running big images.')
end
%%
clear all; close all;

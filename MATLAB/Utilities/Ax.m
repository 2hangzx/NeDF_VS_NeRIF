function [ projections ] = Ax(flow, geo, varargin )
%   Ax(flow, geo) computes projections for flow and geometry information
%   Ax(flow, geo,OPT,VAL,...) uses options and values for computing. The
%   possible options in OPT are:
%   'diffD':    Sets the flow gradient direction. diffD is 'none' by default,
%               indicating the projection is based on flow rather than flow
%               gradient. Otherwise, diffD can be 'x', 'y', 'z'.
%
%   'ptype':    Sets the projection operator type. ptype is 'Siddon' by
%               default.
%   
%   'gpuname':  Sets the name of choised Gpu. Default is all avaliable Gpu. 
%   
%   'Savemat':  With an string in VAL, saves the flow as .mat with
%               VAL as filename
%   'conductmex': conductmex = 0, do not conduct the Ax mex function.
%                 conductmex = 1, conduct the Ax mex function.
%                 Default 1
%--------------------------------------------------------------------------
% This file is part of the BOSLAB Toolbox
% 
% Copyright (c) 2015, Beihang University
%                     All rights reserved.
%
% License:            Open Source under BSD. 
%                     See the full license at
%                     https:XXXXX
%
% Contact:            Weihu22@buaa.edu.cn
% Codes:              https:XXXXX
% Coded by:           Wei Hu
%--------------------------------------------------------------------------
%% Optionals

opts=     {'diffd','ptype','gpuids','savemat','conductmex','blur'};
defaults= [   1  ,  1  ,    1    ,    1 ,  1 , 1 ];

% Check inputs
nVarargs = length(varargin);
if mod(nVarargs,2)
    error('BOSLAB:Ax:InvalidInput','Invalid number of inputs')
end

% check if option has been passed as input
for ii=1:2:nVarargs
    ind=find(ismember(opts,lower(varargin{ii})));
    if ~isempty(ind)
       defaults(ind)=0; 
    else
       error('BOSLAB:Ax:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
    end
end


for ii=1:length(opts)
    opt=opts{ii};
    default=defaults(ii);
    % if one option isnot default, then extranc value from input
    if default==0
        ind=double.empty(0,1);jj=1;
        while isempty(ind)
            ind=find(isequal(opt,lower(varargin{jj})));
            jj=jj+1;
        end
        if isempty(ind)
           error('BOSLAB:Ax:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
        end
        val=varargin{jj};
    end
    
    switch opt
% % % % % %         %diffD
        case 'diffd'
            if default || lower(val)=='n'
                diffselect = 0;
            else
                if ~ischar(val)
                    error('BOSLAB:Ax:InvalidInput','Invalid diffD')
                end
                
                if lower(val)=='z'
                    diffselect=3;
                end
                if lower(val)=='y'
                    diffselect=2;
                end
                if lower(val)=='x'
                    diffselect=1;
                end
                if lower(val)=='d'
                    diffselect=4;
                end
            end
% % % % % %         %ptype 
        case 'ptype'
            if default
                ptype='Siddon';
            else
                expectedProjectionTypes = {'Siddon','interpolated','EF-interpolated','RK-interpolated'}; %'EF-interpolated','RK-interpolated' may have some problems in blur ray projection
                if ~any(strcmp(val, expectedProjectionTypes))
                    error('BOSLAB:Ax:InvalidInput','Invalid ptype')
                end
                ptype=val;
            end
% % % % % %         % gpuidsname
        case 'gpuids'
            if default
                gpuids = GpuIds();
            else
                gpuids = val;
            end

% % % % % %         % do you want to save input data as mat? The mat works 
%                     as input data for Ax_mat.cpp to help debug c++ file.
        case 'savemat'
            if default
                savemat=0;
            else
                savemat=1;
                if ~ischar(val)
                   error('BOSLAB:Ax:InvalidInput','filename is not character')
               end
               filename=val;
            end
            
% % % % % % %         % do you want to conduct Ax_mex function?
        case 'conductmex'
            if default
               conductmex=1;
            else
                if val == 0 || val == 1
                   conductmex=val;  
                else
                    error('BOSLAB:Ax:InvalidInput','conductmex is not 0 or 1')
                end
            end
 % % % % % % %         % do you want to conduct supersamping function?
        case 'blur'
            if default
               blur.flag = 0;
               blur.num = 0;
               blur.delta = [];               
            else                
               blur = val;
            end           
        otherwise
          error('BOSLAB:Ax:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in Ax()']);
    end
end
%% flow
assert(isa(flow,'single'),'BOSLAB:Ax:InvalidInput','Image should be single type');
assert(isreal(flow),'BOSLAB:Ax:InvalidInput','Image should be real (non-complex)');
%% geometry
geo=checkGeo(geo);
assert(isequal([size(flow,1) size(flow,2) size(flow,3)],squeeze(geo.nVoxel.')),'BOSLAB:Ax:BadGeometry','nVoxel does not match with provided image size');
%% Gpu
GpuDevices = gpuids.devices;
%% savemat
if savemat    
    save(filename, 'flow','geo','diffselect','ptype','GpuDevices');
end
%%  call the mex fucntion
if conductmex
    blur0.flag = 0;
    blur0.num = 0;
    blur0.delta = []; 
    projections=Ax_mex(flow,geo,diffselect, ptype, GpuDevices, blur0);

    if blur.flag == 1
        for i = 1: 1: blur.num
            blur_step.flag = 1;
            blur_step.num = 1;
            blur_step.delta = blur.delta(:, i:blur.num:end);
            projections_temp = Ax_mex(flow,geo,diffselect, ptype, GpuDevices, blur_step);
            projections = projections + projections_temp;

        end   
        projections = projections./(blur.num+1);
    end
else
    projections=[];
end
end




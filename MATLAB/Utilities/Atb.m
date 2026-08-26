function [ img ] = Atb( projections,geo,varargin )
%ATB CUDA backprojection operator
%   Atb(projections, geo) computes backprojections for the flow point that 
%                         every rays pass
%   Atb(projections, geo, OPT,VAL,...) uses options and values for computing. 
%   The possible options in OPT are:
%
%   'ptype':    Sets the projection type. ptype is 'matched' by
%               default.
%   
%   'gpuname':  Sets the name of choised Gpu. Default is all avaliable Gpu. 
%   
%   'Savemat':  With an string in VAL, saves the flow as .mat with
%               VAL as filename
%   'conductmex': conductmex = 0, do not conduct the Ax mex function.
%                 conductmex = 1, conduct the Ax mex function.
%                 Default 1
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
%% OPtionals
opts=     {'diffd','ptype','gpuids','savemat','conductmex'};
defaults= [   1  ,  1  ,    1    ,    1   ];
% Check inputs
nVarargs = length(varargin);
if mod(nVarargs,2)
    error('BOSLAB:Atb:InvalidInput','Invalid number of inputs')
end

% check if option has been passed as input
for ii=1:2:nVarargs
    ind=find(ismember(opts,lower(varargin{ii})));
    if ~isempty(ind)
       defaults(ind)=0; 
    else
       error('BOSLAB:Atb:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
    end
end


ninput=1;
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
           error('BOSLAB:Atb:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
        end
        val=varargin{jj};
    end
    
    switch opt
% % % % % %         %diffD
        case 'diffd'
            if default
                diffselect = 0;
            else
                if ~ischar(val)
                    error('BOSLAB:plotFlows:InvalidInput','Invalid diffD')
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
            end
% % % % % %         %ptype 
        case 'ptype'
            if default
                ptype='matched';
            else
                expectedProjectionTypes = {'mask','FDK','matched'};
                if ~any(strcmp(val, expectedProjectionTypes))
                    error('BOSLAB:Atb:InvalidInput','Invalid ptype')
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
                   error('BOSLAB:Atb:InvalidInput','filename is not character')
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
                    error('BOSLAB:Atb:InvalidInput','conductmex is not 0 or 1')
                end
            end
            
        otherwise
          error('BOSLAB:Atb:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in plotFlow()']);
    end
end
%% image
assert(isa(projections,'single'),'BOSLAB:Atb:InvalidInput','Image should be single type');
assert(isreal(projections),'BOSLAB:Atb:InvalidInput','Image should be real (non-complex)');
assert(size(projections,2)>1,'BOSLAB:Atb:InvalidInput', 'Projections should be 2D'); %TODO: needed? 
% assert(size(projections,3)==geo.numCam,'BOSLAB:Atb:InvalidInput', 'Number of projections should match number of angles.'); 
%% geometry
geo=checkGeo(geo);
assert(isequal([size(projections,2) size(projections,1)],[max(geo.nCam(1,:)) max(geo.nCam(2,:))]),'BOSLAB:checkGeo:BadGeometry','nDetector does not match with provided image size');
%% Gpu
GpuDevices = gpuids.devices;
%% savemat
if savemat    
    save(filename, 'projections','geo','diffselect','ptype','GpuDevices');
end
%% Thats it, lets call the mex fucntion
%%  call the mex fucntion
if conductmex
    img = Atb_mex(projections,geo,diffselect, ptype, GpuDevices);
else
    img = [];
end

end

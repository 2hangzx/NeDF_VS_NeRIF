function [x,resL2,qualMeasOut]= CGLS(proj,geo,niter,varargin)
% CGLS solves the BOST problem using the conjugate gradient least
% squares
%
%  CGLS(PROJ,GEO,ANGLES,NITER) solves the reconstruction problem
%   using the projection data PROJ, corresponding
%   to the geometry descrived in GEO, using NITER iterations.
%
%  CGLS(PROJ,GEO,NITER,OPT,VAL,...) uses options and values for solving. The
%   possible options in OPT are:
%
%
%  'Init'    Describes diferent initialization techniques.
%             * 'none'     : Initializes the image to zeros (default)
%             * 'multigrid': Initializes image by solving the problem in
%                            small scale and increasing it when relative
%                            convergence is reached.
%             * 'image'    : Initialization using a user specified
%                            image. Not recomended unless you really
%                            know what you are doing.
%
%  'groundTruth'  an image as grounf truth, to be used if quality measures
%                 are requested, to plot their change w.r.t. this known
%                 data.
%  'restart'  true or false. By default the algorithm will restart when
%             loss of ortogonality is found. 
%--------------------------------------------------------------------------
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

%%

[x,diffselect,Axptype,QualMeasOpts,gpuids,gt,restart,flowmask]=parse_inputs(proj,geo,varargin);

measurequality=~isempty(QualMeasOpts) | ~any(isnan(gt(:)));
if ~any(isnan(gt(:)))
    QualMeasOpts{end+1}='error_norm';
    x0=gt;
    clear gt
end
if nargout<3 && measurequality
    warning("Image metrics requested but none catched as output. Call the algorithm with 3 outputs to store them")
    measurequality=false;
end
qualMeasOut=zeros(length(QualMeasOpts),niter);

resL2=zeros(1,niter);

% //doi: 10.1088/0031-9155/56/13/004
iter=0;
remember=0;
while iter<niter
    r=proj-AxVal_fun(diffselect, x, geo,Axptype,gpuids);    
    p=AtbVal_fun(diffselect, r, geo,gpuids);

    gamma=norm(p(:),2)^2;
    for ii=iter:niter
        iter=iter+1;
        if measurequality && ~strcmp(QualMeasOpts,'error_norm')
            x0 = x; % only store if necesary
        end
        if (iter==1);tic;end
        
        q=AxVal_fun(diffselect, p, geo,Axptype,gpuids); 
        alpha=gamma/norm(q(:),2)^2;
        x=x+alpha*p;
        if ~isempty(flowmask)
            x = x.*flowmask;
        end
        
        
        if measurequality
            qualMeasOut(:,iter)=Measure_Quality(x0,x,QualMeasOpts);
        end
        
        % The following should never happen, but the reallity is that if we use
        % the residual from the algorithm, it starts diverging from this explicit residual value.
        % This is an interesting fact that I believe may be caused either by
        % the mismatch of the backprojection w.r.t the real adjoint, or
        % numerical issues related to doing several order of magnitude
        % difference operations on single precission numbers.
        aux=proj-AxVal_fun(diffselect, x, geo,Axptype,gpuids); 
        resL2(iter)=im3Dnorm(aux,'L2');
        if iter>1 && resL2(iter)>resL2(iter-1)
            % we lost orthogonality, lets restart the algorithm unless the
            % user asked us not to. 
            
            % undo bad step. 
            x=x-alpha*p;
            if ~isempty(flowmask)
                x = x.*flowmask;
            end
            % if the restart didn't work. 
            if remember==iter || ~restart
                disp(['Algorithm stoped in iteration ', num2str(iter),' due to loss of ortogonality.'])
                return;
            end
            remember=iter;
            iter=iter-1;
            disp(['Orthogonality lost, restarting at iteration ', num2str(iter) ])
            break  
            
        end
        % If step is adecuate, then continue withg CGLS
        r=r-alpha*q;
        s=AtbVal_fun(diffselect, r, geo,gpuids);
        gamma1=norm(s(:),2)^2;
        beta=gamma1/gamma;
        gamma=gamma1;
        p=s+beta*p;
        
        
        if (iter==1)
            expected_time=toc*niter;
            disp('CGLS');
            disp(['Expected duration   :    ',secs2hms(expected_time)]);
            disp(['Expected finish time:    ',datestr(datetime('now')+seconds(expected_time))]);
            disp('');
        end
    end
end
end


%% parse inputs'
function [x,diffselect,Axptype,QualMeasOpts,gpuids,gt,restart,flowmask]=parse_inputs(proj,geo,argin)
opts=     {'init','diffd','axptype','qualmeas','gpuids','groundtruth','restart','flowmask'};
defaults=ones(length(opts),1);
initwithimage=0;
% Check inputs
nVarargs = length(argin);
if mod(nVarargs,2)
    error('BOSLAB:CGLS:InvalidInput','Invalid number of inputs')
end

% check if option has been passed as input
for ii=1:2:nVarargs
    ind=find(ismember(opts,lower(argin{ii})));
    if ~isempty(ind)
        defaults(ind)=0;
    else
        error('BOSLAB:CGLS:InvalidInput',['Optional parameter "' argin{ii} '" does not exist' ]);
    end
end

for ii=1:length(opts)
    opt=opts{ii};
    default=defaults(ii);
    % if one option isnot default, then extranc value from input
    if default==0
        ind=double.empty(0,1);jj=1;
        while isempty(ind)
            ind=find(isequal(opt,lower(argin{jj})));
            jj=jj+1;
        end
        if isempty(ind)
            error('BOSLAB:CGLS:InvalidInput',['Optional parameter "' argin{jj} '" does not exist' ]);
        end
        val=argin{jj};
    end
    
    switch opt
        case 'init'
            x=[];
            if default || strcmp(val,'none')
                x=zeros(geo.nVoxel','single');
                continue;
            end
%             if strcmp(val,'multigrid')
%                 x=init_multigrid(proj,geo,angles);
%                 continue;
%             end
            if strcmp(val,'image')
                initwithimage=1;
                continue;
            end
            if isempty(x)
                error('BOSLAB:CGLS:InvalidInput','Invalid Init option')
            end
            %  =========================================================================
        case 'diffd'
            if default || strcmp(lower(val),'n')
                diffselect = 0;
            elseif strcmp(lower(val),'xyz')
                diffselect = 4;
            elseif strcmp(lower(val),'xyzn')
                diffselect = 5;
            elseif strcmp(lower(val),'unified')
                diffselect = 6;
            elseif strcmp(lower(val),'unified-n')
                diffselect = 7;
            elseif strcmp(lower(val),'unified-xyz')
                diffselect = 8;
            elseif strcmp(lower(val),'unified-xyzn')
                diffselect = 9;
            else
                error('BOSLAB:CGLS:InvalidInput','Invalid diffD')
            end

            %  =========================================================================
        case 'axptype'
             if default || strcmp(val,'none')
                Axptype='Siddon';
                continue;
             end
             expectedProjectionTypes = {'Siddon','interpolated','EF-interpolated','RK-interpolated'};
             if any(strcmp(val, expectedProjectionTypes))
                Axptype=val;
                continue;
             else
                 error('BOSLAB:CGLS:InvalidInput','Invalid ptype')
             end
            %  =========================================================================
       case 'flowmask'
             if default 
                flowmask=[];
                continue;
             else
                flowmask=val;
                x = x.* flowmask;
             end     
            %  =========================================================================
        case 'qualmeas'
            if default
                QualMeasOpts={};
            else
                if iscellstr(val)
                    QualMeasOpts=val;
                else
                    error('BOSLAB:CGLS:InvalidInput','Invalid quality measurement parameters');
                end
            end   
        case 'gpuids'
            if default
                gpuids = GpuIds();
            else
                gpuids = val;
            end
        case 'groundtruth'
            if default
                gt=nan;
            else
                gt=val;
                if initwithimage
                    x=gt;
                end
            end
        case 'restart'
            if default
                restart=true;
            else
                restart=val;
            end
        otherwise
            error('BOSLAB:CGLS:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in CGLS()']);
    end
end


end




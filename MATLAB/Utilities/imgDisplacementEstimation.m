function [estimatedU, estimatedV] = imgDisplacementEstimation(iref, idis, geo, varargin)
    opts=     {'initialflow','pyrscale','levels','winsize','iterations','polyn','polysigma','gaussian','pyversion'};
    defaults= [   1  ,           1  ,       1    ,    1 ,       1,        1,        1,         1 ,     1 ];
    
    % Check inputs
    nVarargs = length(varargin);
    if mod(nVarargs,2)
        error('BOSLAB:imgDisplacementEstimation:InvalidInput','Invalid number of inputs')
    end
    
    % check if option has been passed as input
    for ii=1:2:nVarargs
        ind=find(ismember(opts,lower(varargin{ii})));
        if ~isempty(ind)
           defaults(ind)=0; 
        else
           error('BOSLAB:imgDisplacementEstimation:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
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
               error('BOSLAB:imgDisplacementEstimation:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
            end
            val=varargin{jj};
        end
        
        switch opt
    % % % % % %         %initialflow
            case 'initialflow'
                if default
                    initialflow = [];
                else
                    initialflow = val;
                end
    % % % % % %         %pyrscale 
            case 'pyrscale'
                if default
                    pyrscale = 0.5;
                else
                    pyrscale = val;
                end
    % % % % % %         %levels
            case 'levels'
                if default
                    levels = 5;
                else
                    levels = val;
                end
    
    % % % % % %         % winsize
            case 'winsize'
                if default
                    winsize = 16;
                else
                    winsize = val;
                end
                
    % % % % % % %         % iterations
            case 'iterations'
                if default
                   iterations = 5;
                else
                   iterations = val;
                end

     % % % % % % %         % iterations
            case 'polyn'
                if default
                   polyn = 5;
                else
                   polyn = val;
                end  

    % % % % % % %         % iterations
            case 'polysigma'
                if default
                   polysigma = 1.1;
                else
                   polysigma = val;
                end 

    % % % % % % %         % iterations
            case 'gaussian'
                if default
                   gaussian = false;
                else
                   gaussian = val;
                end 
% % % %         % pyversion
            case 'pyversion'
                if default
                   pyversion_flag = false;
                else
                   pyversion_flag = val;
                end 

            otherwise
              error('BOSLAB:imgDisplacementEstimation:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in imgDisplacementEstimation()']);
        end
    end
    %%
    clear estimatedU estimatedV
    for i = 1: 1: geo.numCam
        if ndims(iref)==3 & ndims(idis)==3
            if pyversion_flag
                pycv = py.importlib.import_module('cv2');

                iref_gray = im2uint8(iref(:,:,1));
                idis_gray = im2uint8(idis(:,:,1));
                
                IREF_np = py.numpy.array(im2uint8(iref(:,:,i)));
                IDIS_np = py.numpy.array(im2uint8(idis(:,:,i)));
                
                pyflow = pycv.calcOpticalFlowFarneback( ...
                    py.numpy.array(im2uint8(iref(:,:,i))), py.numpy.array(im2uint8(idis(:,:,i))), py.numpy.array(single(initialflow)), ...
                    py.float(pyrscale), py.int(levels), py.int(winsize), py.int(iterations), py.int(polyn), polysigma, py.int(gaussian));                
                flow = double(pyflow);            
            else
                try
                    flow = cv.calcOpticalFlowFarneback(im2uint8(iref(:,:,i)), im2uint8(idis(:,:,i)),'InitialFlow',initialflow,'PyrScale', pyrscale,...
                    'Levels', levels, 'WinSize', winsize, 'Iterations',iterations, 'PolyN',polyn, 'PolySigma',polysigma,'Gaussian',gaussian);
                catch ME
                    warning('cv.calcOpticalFlowFarneback failed: %s', ME.message);
                    warning('Using zero displacement for camera %d.', i);
                    flow = zeros([size(iref(:,:,i)), 2]);
                end
            end
        else
            if pyversion_flag
                pycv = py.importlib.import_module('cv2');

                iref_gray = im2uint8(iref(:,:,1));
                idis_gray = im2uint8(idis(:,:,1));

                IREF_np = py.numpy.array(im2uint8(iref(:,:,i)));
                IDIS_np = py.numpy.array(im2uint8(idis(:,:,i)));

                pyflow = pycv.calcOpticalFlowFarneback( ...
                    py.numpy.array(im2uint8(iref)), py.numpy.array(im2uint8(idis)), py.numpy.array(single(initialflow)), ...
                    py.float(pyrscale), py.int(levels), py.int(winsize), py.int(iterations), py.int(polyn), polysigma, py.int(gaussian));
                flow = double(pyflow);
            else
                try
                    flow = cv.calcOpticalFlowFarneback(im2uint8(iref), im2uint8(idis),'InitialFlow',initialflow,'PyrScale', pyrscale,...
                    'Levels', levels, 'WinSize', winsize, 'Iterations',iterations, 'PolyN',polyn, 'PolySigma',polysigma,'Gaussian',gaussian);
                catch ME
                    warning('cv.calcOpticalFlowFarneback failed: %s', ME.message);
                    warning('Using zero displacement.');
                    flow = zeros([size(iref), 2]);
                end
            end
        end
        estimatedU(:,:,i) = flow(:,:,1);
        estimatedV(:,:,i) = flow(:,:,2);
    end
end
function pygeo = pyramidGeometry(geo, varargin)
    %% Optionals
    opts=     {'pylevel','pyscale','broaden'};
    defaults= [   1  ,  1 , 1 ];
    
    % Check inputs
    nVarargs = length(varargin);
    if mod(nVarargs,2)
        error('BOSLAB:pyramidGeometry:InvalidInput','Invalid number of inputs')
    end
    
    % check if option has been passed as input
    for ii=1:2:nVarargs
        ind=find(ismember(opts,lower(varargin{ii})));
        if ~isempty(ind)
           defaults(ind)=0; 
        else
           error('BOSLAB:pyramidGeometry:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
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
               error('BOSLAB:pyramidGeometry:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
            end
            val=varargin{jj};
        end
        
        switch opt
    % % % % % %         %pylevel
            case 'pylevel'
                if default
                    pylevel = 1;
                else
                    if val ~= fix(val) || val <= 0
                        error('BOSLAB:pyramidGeometry:InvalidInput','pylevel has to be positive integer');
                    else
                        pylevel = val;
                    end
                end
    % % % % % %         %ptype 
            case 'pyscale'
                if default
                    pyscale = 0.5;
                else
                    if ~isnumeric(val) || val == 0
                        error('BOSLAB:pyramidGeometry:InvalidInput','pyscale has to be numetric & != 0')
                    else
                        pyscale = val;
                    end
                end
     % % % % % %         %broaden 
            case 'broaden'
                if default
                    broaden = 0.5;
                else
                    
                    broaden = val;
                end            
            otherwise
              error('BOSLAB:pyramidGeometry:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in pyramidGeometry()']);
        end
    end
    %%    
    for pyi = 1:1:pylevel
        if pyi == 1
            [geo.UVROI, geo.ROICorner, geo.raymask] = intersectionPoint(geo,pyi,broaden);
        else
            geo = pygeo(pyi-1).geo;
            geo.nVoxel = ceil(pygeo(pyi-1).geo.nVoxel*pyscale);
            geo.sVoxel = geo.nVoxel .* geo.dVoxel;
            [geo.UVROI, geo.ROICorner, geo.raymask] = intersectionPoint(geo,pyi,broaden);
        end
        pygeo(pyi).geo = geo;
    end 
%     pygeo(1).geo = geo;
    pygeo(1).pylevel = pylevel;
    pygeo(1).pyscale = pyscale;
end
%% intersection point calculate
function [UVROI, ROICorners,ray_mask_pool] = intersectionPoint(geo,pyi,broaden)
    for i = 1: 1: geo.numCam
        Ocr = geo.Ocr(:, i);
        Opr = geo.Opr;
        sVoxel = geo.sVoxel;
        Pbkcorner = geo.PbkCorner(:,4*(i-1)+1: 4*i);
        %%%%%%%%%%%%%%%%%%%%%%%% eight coner of flow
        flowcorner(:, 1) = [Opr(1) - sVoxel(1)/2; Opr(2) - sVoxel(2)/2; Opr(3) + sVoxel(3)/2];
        flowcorner(:, 2) = [Opr(1) - sVoxel(1)/2; Opr(2) + sVoxel(2)/2; Opr(3) + sVoxel(3)/2];
        flowcorner(:, 3) = [Opr(1) + sVoxel(1)/2; Opr(2) + sVoxel(2)/2; Opr(3) + sVoxel(3)/2];
        flowcorner(:, 4) = [Opr(1) + sVoxel(1)/2; Opr(2) - sVoxel(2)/2; Opr(3) + sVoxel(3)/2];
        flowcorner(:, 5) = [Opr(1) - sVoxel(1)/2; Opr(2) - sVoxel(2)/2; Opr(3) - sVoxel(3)/2];
        flowcorner(:, 6) = [Opr(1) - sVoxel(1)/2; Opr(2) + sVoxel(2)/2; Opr(3) - sVoxel(3)/2];
        flowcorner(:, 7) = [Opr(1) + sVoxel(1)/2; Opr(2) + sVoxel(2)/2; Opr(3) - sVoxel(3)/2];
        flowcorner(:, 8) = [Opr(1) + sVoxel(1)/2; Opr(2) - sVoxel(2)/2; Opr(3) - sVoxel(3)/2];
        %%%%%%%%%%%%%%%%%%%%%%%% intersectionPoint based on mm
        for ci = 1: 1: 8 
            A = Pbkcorner(:,1);
            B = Pbkcorner(:,2);
            C = Pbkcorner(:,3);
            AB = B-A;
            AC = C-A;
            nv = cross(AB, AC);
            d = sum(nv.*A);
            P1 = Ocr;
            P2 = flowcorner(:, ci);
            t = ( d - sum(nv.*Ocr) ) / sum(nv.*(P2-P1));
            PIpoint(:, ci) = P1 + t*(P2-P1);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%  four corners
        for ci = 1: 1: 4           
            [~, idx] = min( sqrt(sum( (PIpoint- Pbkcorner(:,ci)).^2 , 1)) );
            PIcorner(:,ci) = PIpoint(:, idx);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%  intersectionPoint based on pixel
        K = geo.IMCam(:,3*(i-1)+1: 3*i);
        R = geo.RrCam(:,3*(i-1)+1: 3*i);
        T = geo.TrCam(:, i);
        Zbc = geo.Zbc(i);
        uv1 = K*( R*PIcorner+T )/Zbc;        
        uv1 = round(uv1);        
        if any(uv1(1,:)<1-broaden) || any(uv1(1,:)>geo.nCam(1,i)+broaden) || ...
            any(uv1(2,:)<1-broaden) || any(uv1(2,:)>geo.nCam(2,i)+broaden)
            error(['Error pyramidGeometry: the ROI image of camera', num2str(i), ' is large than the nCam in pylevel=',num2str(pyi),'. Please reset the voxel size or the camera size']);
        end
        uv1(1,uv1(1,:)<1)=1;
        uv1(1,uv1(1,:)>geo.nCam(1,i))=geo.nCam(1,i);
        uv1(2,uv1(2,:)<1)=1;
        uv1(2,uv1(2,:)>geo.nCam(2,i))=geo.nCam(2,i);
        
        [valid_mask, valid_inds] = filterRaysTopBottom(Ocr, flowcorner, uv1, R, K, Zbc, T,  geo.nCam(2,i), geo.nCam(1,i));
        ray_mask = zeros(geo.nCam(2,i), geo.nCam(1,i));
        ray_mask(valid_inds) = valid_mask;
        ray_mask_pool(:,:,i) = ray_mask;
        %%%%%%%%%%%%%%%%%%%%%%%%  ROI corner
        ROICorner = R'*(inv(K)*Zbc*uv1-T);
        %%%%%%%%%%%%%%%%%%%%%%%  
        UVROI(:, i) = [min(uv1(1,:)); min(uv1(2,:)); max(uv1(1,:))-min(uv1(1,:))+1;  max(uv1(2,:))-min(uv1(2,:))+1];
        ROICorners(:,4*(i-1)+1: 4*i) = ROICorner;



    end
end

function [valid_mask, inds] = filterRaysTopBottom(Ocr, flowcorner, uv1, R, K, Zbc, T,H,W)

%% 1.Construct AABB from flow field corners
xmin = min(flowcorner(1,:)); xmax = max(flowcorner(1,:));
ymin = min(flowcorner(2,:)); ymax = max(flowcorner(2,:));
zmin = min(flowcorner(3,:)); zmax = max(flowcorner(3,:));
aabb_min = [xmin; ymin; zmin];
aabb_max = [xmax; ymax; zmax];

%% 2. Generate ROI pixel grid
u_min = round(min(uv1(1,:))); u_max = round(max(uv1(1,:)));
v_min = round(min(uv1(2,:))); v_max = round(max(uv1(2,:)));
[U, V] = meshgrid(u_min:u_max, v_min:v_max);
inds = sub2ind([H, W], V, U);
U = U(:); V = V(:);
num_pixels = length(U);

%% 3. Convert pixels to rays (camera → world)
uv_hom = [U'; V'; ones(1,num_pixels)];
pts_cam = K \ (uv_hom * Zbc);
pts_world = R' * (pts_cam - T);

rays_o = repmat(Ocr, 1, num_pixels);
rays_d = pts_world - rays_o;

%% 4. Ray-AABB intersection (slab method)
eps_dir = 1e-8;  
rays_d(abs(rays_d)<eps_dir) = eps_dir;

tmin = (aabb_min - rays_o) ./ rays_d;
tmax = (aabb_max - rays_o) ./ rays_d;

t1 = min(tmin, tmax);
t2 = max(tmin, tmax);

t_near = max(t1,[],1);
t_far  = min(t2,[],1);

hit = (t_far > t_near) & (t_far > 0);

%% 5. Check exit face of the ray
p_far = rays_o + rays_d .* t_far;  % 射线离开box的位置

% 如果射出点在上下表面 → 无效
exit_top_bottom = abs(p_far(2,:) - ymin) < eps_dir | abs(p_far(2,:) - ymax) < eps_dir;

%% 6. Output valid mask
valid = hit & ~exit_top_bottom;
valid_mask = reshape(valid, v_max-v_min+1, u_max-u_min+1);
end
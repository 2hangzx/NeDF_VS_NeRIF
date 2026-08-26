function geo=defaultGeometry(varargin)
%geo = defaultGeometry() generates a default geomtry for tests.
% Optional parameters
%
% 'nVoxel'    : 3x1 matrix of size of the image
% 'angles'    : Nx1 matrix of camera projection angles (in radians)

%% Optionals
opts=     {'nvoxel','svoxel','angles','ncam','dcam','fcam','zbc','zpc','opr'};
defaults= [   1  ,  1  ,    1    ,    1 ,      1,     1,     1,    1,    1  ];

% Check inputs
nVarargs = length(varargin);
if mod(nVarargs,2)
    error('BOSLAB:defaultGeometry:InvalidInput','Invalid number of inputs')
end

% check if option has been passed as input
for ii=1:2:nVarargs
    ind=find(ismember(opts,lower(varargin{ii})));
    if ~isempty(ind)
       defaults(ind)=0; 
    else
       error('BOSLAB:defaultGeometry:InvalidInput',['Optional parameter "' varargin{ii} '" does not exist' ]); 
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
           error('BOSLAB:defaultGeometry:InvalidInput',['Optional parameter "' varargin{jj} '" does not exist' ]); 
        end
        val=varargin{jj};
    end
    
    switch opt
% % % % % %         %'nVoxel
        case 'nvoxel'
            if default
                nVoxel = [128;128;128];
            else
                nVoxel = val(:);
            end
% % % % % %         %'sVoxel
        case 'svoxel'
            if default
                sVoxel = [24;24;24];
            else
                sVoxel = val(:);
            end
% % % % % %         % angles
        case 'angles'
            if default
                angles = linspace(18, 162, 9);
            else
                angles = val(:)';
            end

% % % % % %         % nCam
        case 'ncam'
            if default
                nCam = [1280;1024];
            else
                nCam = val(:);
            end
            
% % % % % % %         % dCam
        case 'dcam'
            if default
               dCam = [0.0048;0.0048];
            else
               dCam = val(:);
            end
% % % % % % %         % fCam
        case 'fcam'
            if default
               fCam = 50;
            else
               fCam = val;
            end
% % % % % % %         % Zbc
        case 'zbc'
            if default
               Zbc = 1101;
            else
               Zbc = val;
            end  
% % % % % % %         % Zpc
        case 'zpc'
            if default
               Zpc = 546;
            else
               Zpc = val;
            end   
% % % % % % %         % Opr
        case 'opr'
            if default
               Opr = [0;0;0];
            else
               Opr = val(:);
            end  
        otherwise
          error('BOSLAB:defaultGeometry:InvalidInput',['Invalid input name:', num2str(opt),'\n No such option in defaultGeometry()']);
    end
end
if Zpc > Zbc
    error('the distance between probe area and optical center should be smaller than the disctance between background and optical center');
end
%% Example
% VARIABLE                                   DESCRIPTION                               UNITS
% probe area
geo.nVoxel = nVoxel;                        % number of voxels                          (vx)
geo.sVoxel = sVoxel;                        % size of each voxel                        (mm)
geo.dVoxel = geo.sVoxel./geo.nVoxel;        % size of each voxel                        (mm)
geo.Opr = Opr;                              % probe area center position                (mm)
% camera & background
geo.numCam = size(angles, 2);				% number of cameras             
geo.nCam = repmat(nCam,1,geo.numCam);       % number of pixels                          (px)     
geo.dCam = repmat(dCam,1,geo.numCam);   
                                            % size of each pixel                        (mm)                                           
geo.sCam = geo.nCam.*geo.dCam;              % total size of the detector                (mm)
geo.fCam = repmat(fCam,1,geo.numCam);       % focal length                              (mm)
geo.Zbc = repmat(Zbc,1,geo.numCam);         % distance between bk and cam center        (mm)
geo.Zpc = repmat(Zpc,1,geo.numCam);         % distance between probe area center and cam center        
                                            %                                           (mm)
geo.fd = 1./(1./geo.fCam - 1./geo.Zbc);     % distance between img plane and cam center (mm) 

% calculate camera parameter in the individual camera coordinate system
for i = 1: 1: geo.numCam
    geo.IMCam(:, 3*(i-1)+1: 3*i) = [geo.fd(i)/geo.dCam(1,i) 0 geo.nCam(1,i)/2;
                                    0 geo.fd(i)/geo.dCam(2,i) geo.nCam(2,i)/2;
                                    0 0 1];           
                                            % intrinsic matrix
end
geo.RCam = repmat(eye(3),1,geo.numCam);     % individual rotation matrix
geo.TCam = repmat(zeros(3,1),1,geo.numCam); % individual translation matrix

% calculate camera relative parameter in the reference camera coordinate system
geo.RrCam = zeros(3, geo.numCam*3);         % reference rotation matrix
geo.TrCam = zeros(3, geo.numCam);           % reference translation matrix
for i = 1: 1: geo.numCam
    Oc = [0;0;0];
    Op = [geo.Zpc(i)*sind(angles(i));0;geo.Zpc(i)*cosd(angles(i))];
    theta = angles(i)/180*pi;%atan( Op(1)/ Op(3) )
    RrCam = [cos(theta) 0 sin(theta);
             0 1 0;
            -sin(theta) 0 cos(theta); ]'; 
    TrCam = Op;
    geo.RrCam(:, 3*(i-1)+1: 3*i) = RrCam;    % reference rotation matrix
    geo.TrCam(:, i) = RrCam*TrCam+geo.RrCam(:, 3*(i-1)+1: 3*i)*geo.Opr;      % reference translation matrix
%     geo.Ocr(:,i) = Oc - Op;
end
% calculate camera relative parameter in the reference camera coordinate system
for i = 1: 1: geo.numCam    
    geo.Ocr(:,i) = -geo.RrCam(:, 3*(i-1)+1: 3*i)'*geo.TrCam(:, i);
end
geo.Opr = geo.Opr - geo.Opr;
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
end


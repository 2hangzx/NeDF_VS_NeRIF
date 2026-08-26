function geo=checkGeo(geo)
% CHECKGEO checks for correctness in the geometry struct and prepares it
% for usage in the mex files. 
geofields_mandatory={'nVoxel','sVoxel','dVoxel','Opr',...
    'numCam','nCam', 'dCam', 'sCam','fCam','Zbc','Zpc','fd',...
    'IMCam','RCam','TCam','RrCam','TrCam','Ocr',...
    'PbkCorner'};

geofields_optional={'accuracy','Ix','Iy','UVROI','ROICorner'};
allfields=horzcat(geofields_mandatory,geofields_optional);
fnames=fieldnames(geo);
% Find if geo has fields we do not recongize
unknown=~ismember(fnames,allfields);
% there must be not unknown variables
% TODO: Do we really want to enforce this? Perhaps just a warning?
assert(~sum(unknown),'BOSLAB:checkGeo:BadGeometry',['The following fields are not known by BOSLAB:\n' strjoin(fnames(unknown)),'\nMake sure you have not misspelled any field or introduced unnecesary fields.'])



% check mandatory fields
mandatory=ismember(geofields_mandatory,fnames);
assert(sum(mandatory)==length(geofields_mandatory),'BOSLAB:checkGeo:BadGeometry',['The following fields are missing:\n' strjoin(geofields_mandatory(~mandatory))])

% Lets check the validity of them now. We need to be explicit in there

%check if they are double
for ii=1:length(geofields_mandatory)
    assert(isa(geo.(geofields_mandatory{ii}),'double'),'BOSLAB:checkGeo:BadGeometry',['Field geo.', geofields_mandatory{ii},' is not double type.'])
end

% flow data
assert(isequal(size(geo.nVoxel),[3 1]),'BOSLAB:checkGeo:BadGeometry','geo.nVoxel should be 3x1')
assert(isequal(geo.nVoxel,round(geo.nVoxel)),'BOSLAB:checkGeo:BadGeometry','geo.nVoxel should be a natural number.')

assert(isequal(size(geo.sVoxel),[3 1]),'BOSLAB:checkGeo:BadGeometry','geo.sVoxel should be 3x1')

assert(isequal(size(geo.dVoxel),[3 1]),'BOSLAB:checkGeo:BadGeometry','geo.sVoxel should be 3x1')

assert(sum(abs(geo.dVoxel.*geo.nVoxel-geo.sVoxel))<1e-6, 'BOSLAB:checkGeo:BadGeometry', 'nVoxel*dVoxel is not sVoxel, something is wrong in the numbers')

assert(isequal(size(geo.Opr),[3 1]),'BOSLAB:checkGeo:BadGeometry','geo.Opr should be 3x1')

% Camera data
assert(isscalar(geo.numCam),'BOSLAB:checkGeo:BadGeometry','geo.numCam should be scalar')

assert(isequal(size(geo.nCam),[2 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.nCam should be 2x',num2str(geo.numCam))

assert(isequal(size(geo.dCam),[2 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.dCam should be 2x',num2str(geo.numCam))

assert(isequal(size(geo.sCam),[2 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.sCam should be 2x',num2str(geo.numCam))

assert(isequal(geo.nCam,round(geo.nCam)),'BOSLAB:checkGeo:BadGeometry','geo.nCam should be a natural number.')

assert(sum(sum(abs(geo.dCam.*geo.nCam-geo.sCam)))<1e-6, 'BOSLAB:checkGeo:BadGeometry', 'nCam*dCam is not sCam, something is wrong in the numbers')

assert(isequal(size(geo.fCam),[1 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.nCam should be 1x',num2str(geo.numCam))

assert(isequal(size(geo.Zbc),[1 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.Zbc should be 1x',num2str(geo.numCam))

assert(isequal(size(geo.Zpc),[1 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.Zpc should be 1x',num2str(geo.numCam))

assert(all(geo.Zbc>=geo.Zpc), 'TIGRE:checkGeo:BadGeometry','Zbc should be bigger to Zpc');

assert(isequal(size(geo.fd),[1 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.fd should be 1x',num2str(geo.numCam))

assert(isequal(size(geo.IMCam),[3 3*geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.IMCam should be 3x',num2str(3*geo.numCam))

assert(isequal(size(geo.RCam),[3 3*geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.RCam should be 3x',num2str(3*geo.numCam))

assert(isequal(size(geo.TCam),[3 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.TCam should be 3x',num2str(geo.numCam))

assert(isequal(size(geo.RrCam),[3 3*geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.RrCam should be 3x',num2str(3*geo.numCam))

assert(isequal(size(geo.TrCam),[3 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.TrCam should be 3x',num2str(geo.numCam))

assert(isequal(size(geo.Ocr),[3 geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.Ocr should be 3x',num2str(geo.numCam))

% background
assert(isequal(size(geo.PbkCorner),[3 4*geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.PbkCorner should be 3x',num2str(4*geo.numCam))

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we know that optional fields are properly written or they would have
% been flagged before.
% We need to be explicit here again.
% geofields_optional={'offOrigin','offDetector','rotDetector','COR',...
%                     'mode','accuracy'};


if isfield(geo,'accuracy')
    assert(isscalar(geo.accuracy),'BOSLAB:checkGeo:BadGeometry','geo.accuracy should be a scalar');
    assert(isa(geo.accuracy,'double'),'BOSLAB:checkGeo:BadGeometry','geo.accuracy should be double');
    if geo.accuracy>1
        warning('geo.accuracy too big, you will ignore image information resulting in wrong reconstruction.\n Change geo.accuracy to smaller or equal than 1.')
    end
else
    geo.accuracy=0.5;
end

% if isfield(geo,'Ix') 
%     assert(isequal(size(geo.Ix),[max(geo.nCam(2,:)) max(geo.nCam(1,:)) geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.Ix should be ',num2str(max(geo.nCam(2,:))), 'x',num2str(max(geo.nCam(1,:))),'x',geo.numCam);   
% end
% 
% if isfield(geo,'Iy') 
%     assert(isequal(size(geo.Iy),[max(geo.nCam(2,:)) max(geo.nCam(1,:)) geo.numCam]),'BOSLAB:checkGeo:BadGeometry','geo.Iy should be ',num2str(max(geo.nCam(2,:))), 'x',num2str(max(geo.nCam(1,:))),'x',geo.numCam);   
% end
end


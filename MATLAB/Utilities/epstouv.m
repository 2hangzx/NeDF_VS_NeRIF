function [projectionsU, projectionsV] =epstouv(geo, projectionsX, projectionsY, projectionsZ)
    for i=1:1:geo.numCam
        K = geo.IMCam(:, 3*(i-1)+1: 3*i);
        R = geo.RrCam(:, 3*(i-1)+1: 3*i);
        new_R = K*R;%
        projectionsXYZ = cat(3,projectionsX(:,:,i), projectionsY(:,:,i), projectionsZ(:,:,i));
        projectionsXYZ = reshape(projectionsXYZ, size(projectionsXYZ, 1) * size(projectionsXYZ, 2), size(projectionsXYZ, 3));
        Zbc = geo.Zbc(i);
        Zbp = geo.Zbc(i)-geo.Zpc(i);
        uvo = Zbp/Zbc*new_R * projectionsXYZ';
        u = reshape( uvo(1,:), [geo.nCam(2,i) geo.nCam(1,i)]);
        v = reshape( uvo(2,:), [geo.nCam(2,i) geo.nCam(1,i)]);
        o = reshape( uvo(3,:), [geo.nCam(2,i) geo.nCam(1,i)]);
        
        projectionsU(:,:,i)=u;
        projectionsV(:,:,i)=v;               
    end
end
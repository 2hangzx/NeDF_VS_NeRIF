function [projectionsX, projectionsY, projectionsZ] =uvtoeps(geo, projectionsU, projectionsV)
    for i=1:1:geo.numCam
        u = projectionsU(:,:,i);
        v = projectionsV(:,:,i);
        uv0 = [u(:)'; v(:)'; zeros(1, length(u(:)))];
        K = geo.IMCam(:, 3*(i-1)+1: 3*i);
        R = geo.RrCam(:, 3*(i-1)+1: 3*i);
        Zbc = geo.Zbc(i);
        Zbp = geo.Zbc(i)-geo.Zpc(i);
        invKR(:, 3*(i-1)+1: 3*i) = inv(K*R);
        eps = inv(K*R)*Zbc/Zbp*uv0;
        projectionsX(:,:,i) = reshape(eps(1,:), size(u));
        projectionsY(:,:,i) = reshape(eps(2,:), size(u));
        projectionsZ(:,:,i) = reshape(eps(3,:), size(u));
    end
end
function projectionsN = calcuProjectionN(projectionsU, projectionsV, geo)
    projectionsN = zeros(geo.nCam(2),geo.nCam(1),geo.numCam);
    for i = 1:1: geo.numCam
        UVROI = geo.UVROI(:,i);
        u = projectionsU(UVROI(2):UVROI(2)+UVROI(4)-1, UVROI(1):UVROI(1)+UVROI(3)-1,i);
        v = projectionsV(UVROI(2):UVROI(2)+UVROI(4)-1, UVROI(1):UVROI(1)+UVROI(3)-1,i);
        BP = geo.Zbc(i) - geo.Zpc(i);
        BO = geo.Zbc(i);
        dx = 1;
        scale = BO * geo.dCam(1,i)/geo.fd(i);
        eps_x = atan(u*scale./BP);
        eps_y = atan(v*scale./BP);
        Dy = dx*scale*(BO-BP)/BO;
        p = floor(size(u,2)/2);
        m = floor(size(u,1)/2);
        Y_Line = m*Dy:-Dy:-m*Dy;
        X_Line = -p*Dy:Dy:p*Dy;
        X_Line = X_Line(1:size(u,2));
        Y_Line = Y_Line(1:size(u,1));
        
        [X,Y] = meshgrid(X_Line,Y_Line);
        div = divergence(X, Y, eps_x, eps_y);
        h = abs( X(1,2) - X(1,1) );
        
        f = div * h^2;
        f(1, :) = f(1, :) - eps_y(1, :)*h;
        f(end, :) = f(end, :) + eps_y(end, :)*h;
%         f(:,1) = f(:,1) + eps_x(:,1)*h;
%         f(:,end) = f(:, end) - eps_x(:, end)*h;
        
        det_ba_poi = full(poisson_2D( f,[],[],'boundary_condition',{'top','bottom'}));%,'left','right'
        det_ba_poi( det_ba_poi>0 ) = 0;
        det_ba_poi = reshape(det_ba_poi, size(eps_x));

        projectionsN(UVROI(2):UVROI(2)+UVROI(4)-1, UVROI(1):UVROI(1)+UVROI(3)-1,i) = det_ba_poi;
    end    
end
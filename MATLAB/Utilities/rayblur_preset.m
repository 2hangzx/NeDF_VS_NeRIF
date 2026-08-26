function blur = rayblur_preset(geo, r, N, type)
    rng(1234);
    if strcmp(type, 'poisson')
        for i = 1:1:geo.numCam
            P1 = geo.PbkCorner(:,4*(i-1)+1);
            P2 = geo.PbkCorner(:,4*(i-1)+2);
            P3 = geo.PbkCorner(:,4*(i-1)+3);
        
            u = P2 - P1;
            u = u / norm(u);  
            
            normal = cross(P2 - P1, P3 - P1);
            normal = normal / norm(normal);
            
            v = cross(normal, u);  
            v = v / norm(v);
        
            eta = 0.65;  
            dmin = r(i) * sqrt(eta / N);  
            pts2d = poissonDiskCircle(r(i), dmin);
            
            if size(pts2d,1) > N
                pts2d = pts2d(randperm(size(pts2d,1), N), :);
            end
            alpha = pts2d(:,1);
            beta = pts2d(:,2);
            
            deltaX = alpha * u(1) + beta * v(1);
            deltaY = alpha * u(2) + beta * v(2);
            deltaZ = alpha * u(3) + beta * v(3);

            delta(:,N*(i-1)+1:N*i) = [deltaX'; deltaY'; deltaZ'];
        end
    elseif strcmp(type, 'random')
        for i = 1:1:geo.numCam
            P1 = geo.PbkCorner(:,4*(i-1)+1);
            P2 = geo.PbkCorner(:,4*(i-1)+2);
            P3 = geo.PbkCorner(:,4*(i-1)+3);
        
            u = P2 - P1;
            u = u / norm(u);  
            
            normal = cross(P2 - P1, P3 - P1);
            normal = normal / norm(normal);
            
            v = cross(normal, u);  
            v = v / norm(v);
        
            theta = 2*pi*rand(N,1);
            rho = r(i) * sqrt(rand(N,1));   
            alpha = rho .* cos(theta);
            beta  = rho .* sin(theta);
        
            deltaX = alpha * u(1) + beta * v(1);
            deltaY = alpha * u(2) + beta * v(2);
            deltaZ = alpha * u(3) + beta * v(3);

            delta(:,N*(i-1)+1:N*i) = [deltaX'; deltaY'; deltaZ'];
        end

    end
    blur.flag = 1;    
    blur.num = N;
    blur.delta = delta;
end

function pts = poissonDiskCircle(R, rmin)
%  Poisson disk sampling in a circle

cellSize = rmin / sqrt(2);

gridRange = ceil((2*R) / cellSize);
grid = -ones(gridRange, gridRange);
pts = [];
active = [];

theta = 2*pi*rand;
rho = R*sqrt(rand);
p0 = [rho*cos(theta), rho*sin(theta)];

pts = p0;
active = 1;

ix = floor((p0(1)+R)/cellSize) + 1;
iy = floor((p0(2)+R)/cellSize) + 1;
grid(ix, iy) = 1;

while ~isempty(active)
    idx = active(randi(length(active)));
    p = pts(idx,:);
    found = false;

    for i = 1:k
        ang = 2*pi*rand;
        rad = rmin * (1 + rand);
        q = p + rad * [cos(ang), sin(ang)];

        if norm(q) > R
            continue;
        end

        qx = floor((q(1)+R)/cellSize) + 1;
        qy = floor((q(2)+R)/cellSize) + 1;

        if qx < 1 || qy < 1 || qx > gridRange || qy > gridRange
            continue;
        end

        ok = true;
        for iix = max(1,qx-2):min(gridRange,qx+2)
            for iiy = max(1,qy-2):min(gridRange,qy+2)
                pid = grid(iix,iiy);
                if pid ~= -1
                    if norm(q - pts(pid,:)) < rmin
                        ok = false;
                        break;
                    end
                end
            end
            if ~ok, break; end
        end

        if ok
            pts = [pts; q];
            active = [active, size(pts,1)];
            grid(qx,qy) = size(pts,1);
            found = true;
            break;
        end
    end

    if ~found
        active(active == idx) = [];
    end
end
end

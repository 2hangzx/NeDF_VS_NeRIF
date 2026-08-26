function IM=makedotpatternIMG(density,radim, imgsize,varargin)
    S= imgsize(1)*imgsize(2);
    n = ceil((density* S)/(4*(radim+0.5)^2));
    ppp = n/S;
    if nargin > 3 && strcmp(varargin{1}, 'random')
        x = randi([1, imgsize(1)], n, 1);
        y = randi([1, imgsize(2)], n, 1);      
        pts = [x, y];
    elseif  nargin > 3 && strcmp(varargin{1}, 'poisson')
        pts=round(poissonDisc(imgsize,(radim+0.5)*(2.4-0.4*(density-0.4)/0.1),n,1));
    else
        pts=round(poissonDisc(imgsize,(radim+0.5)*(2.4-0.4*(density-0.4)/0.1),n,1));
    end
    pts(pts(:,1)<1,1) =1;
    pts(pts(:,1)>imgsize(1),1) =imgsize(1);
    pts(pts(:,2)<1,2) =1;
    pts(pts(:,2)>imgsize(2),2) =imgsize(2);

    IM = zeros(imgsize)+1;
    for i = 1:1: size(pts,1)
        IM(pts(i,1), pts(i,2)) =0;
    end
    rm = -radim:1:radim;
    for i = 1:1:length(rm)
        pts1 = pts(:,1)+rm(i);
        pts1(pts1<1) =1;
        pts1(pts1>imgsize(1)) =imgsize(1);
        for j = 1:1:length(rm)
            pts2 = pts(:,2)+rm(j);
            pts2(pts2<1) =1;
            pts2(pts2>imgsize(2)) =imgsize(2);           
             for k = 1:1: size(pts,1)
                IM(pts1(k), pts2(k)) =0;
            end
        end
    end
end
function Atbval = AtbVal_fun(diffselect, r, geo,gpuids)
    if diffselect == 0
        Atbval=Atb(r,geo,'conductmex',1,'gpuids',gpuids);
    elseif diffselect == 4
        rx = r(:,:,1: geo.numCam);
        ry = r(:,:,geo.numCam+1: geo.numCam*2);
        rz = r(:,:,geo.numCam*2+1: geo.numCam*3);
        AtbvalX = Atb(rx,geo,'diffD','X','conductmex',1,'gpuids',gpuids);
        AtbvalY = Atb(ry,geo,'diffD','Y','conductmex',1,'gpuids',gpuids);
        AtbvalZ = Atb(rz,geo,'diffD','Z','conductmex',1,'gpuids',gpuids);
        Atbval = AtbvalX+AtbvalY+AtbvalZ;
    elseif diffselect == 5
        rx = r(:,:,1: geo.numCam);
        ry = r(:,:,geo.numCam+1: geo.numCam*2);
        rz = r(:,:,geo.numCam*2+1: geo.numCam*3);
        rn = r(:,:,geo.numCam*3+1: geo.numCam*4);
        AtbvalX = Atb(rx,geo,'diffD','X','conductmex',1,'gpuids',gpuids);
        AtbvalY = Atb(ry,geo,'diffD','Y','conductmex',1,'gpuids',gpuids);
        AtbvalZ = Atb(rz,geo,'diffD','Z','conductmex',1,'gpuids',gpuids);
        AtbvalN = Atb(rn,geo,'conductmex',1);
        Atbval = AtbvalX+AtbvalY+AtbvalZ+AtbvalN;
    elseif diffselect == 6
        rIxIt = geo.Ix .* r;
        rIyIt = geo.Iy .* r;
        for i=1:1:geo.numCam   
            K = geo.IMCam(:, 3*(i-1)+1: 3*i);
            R = geo.RrCam(:, 3*(i-1)+1: 3*i);       
            Zbc = geo.Zbc(i);
            Zbp = geo.Zbc(i)-geo.Zpc(i);
            new_R = Zbp/Zbc*K*R;
            co_rIxItDx(1,1,i) = new_R(1,1);
            co_rIxItDy(1,1,i) = new_R(1,2);
            co_rIxItDz(1,1,i) = new_R(1,3);
        
            co_rIyItDx(1,1,i) = new_R(2,1);
            co_rIyItDy(1,1,i) = new_R(2,2);
            co_rIyItDz(1,1,i) = new_R(2,3);
        end  
        AtbvalX = Atb(co_rIxItDx.*rIxIt + co_rIyItDx.*rIyIt,geo,'diffD','X','conductmex',1);
        AtbvalY = Atb(co_rIxItDy.*rIxIt + co_rIyItDy.*rIyIt ,geo,'diffD','Y','conductmex',1);
        AtbvalZ = Atb(co_rIxItDz.*rIxIt + co_rIyItDz.*rIyIt,geo,'diffD','Z','conductmex',1);

        Atbval = AtbvalX + AtbvalY + AtbvalZ;

    end
end
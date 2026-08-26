function Axval = AxVal_fun(diffselect, x, geo,Axptype,gpuids)
    if diffselect == 0
        Axval=Ax(x, geo,'ptype',Axptype,'conductmex',1,'gpuids',gpuids);
    elseif diffselect == 4
        AxvalX=Ax(x, geo,'diffD','X','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalY=Ax(x, geo,'diffD','Y','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalZ=Ax(x, geo,'diffD','Z','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        Axval = cat(3,AxvalX, AxvalY, AxvalZ);
    elseif diffselect == 5
        AxvalX=Ax(x, geo,'diffD','X','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalY=Ax(x, geo,'diffD','Y','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalZ=Ax(x, geo,'diffD','Z','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalN=Ax(x, geo,'ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        Axval = cat(3,AxvalX, AxvalY, AxvalZ,AxvalN);
    elseif diffselect == 6
        AxvalX=Ax(x, geo,'diffD','X','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalY=Ax(x, geo,'diffD','Y','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        AxvalZ=Ax(x, geo,'diffD','Z','ptype',Axptype,'conductmex',1,'gpuids',gpuids);
        [AxvalU, AxvalV] = epstouv(geo, AxvalX, AxvalY, AxvalZ);
        Axval =  geo.Ix.*AxvalU + geo.Iy.*AxvalV;
    end
end
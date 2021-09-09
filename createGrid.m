function integergrid = createGrid(type,maxShift,gridSpacing)
%CREATEGRID creates a grid with a specific geometry
%   The grid can be 'Cairo', 'Hex', 'Prismatic', 'Square', or 'Tri'
%   The parameter maxShift determines the extension of the grid, and
%   gridSpacing determines the space between two pointd of the grid.

switch type
        case 'Cairo'
            gr = CreateCairoGrid(100,100);
        case 'Hex'
            gr = CreateHexGrid(100,100);
        case 'Prismatic'
            gr = CreatePrismaticGrid(100,100);
        case 'Square'
            gr = CreateSquareGrid(100,100);
        case 'Tri'
            gr = CreateTriGrid(100,100);
end

gr(1,:) = gr(1,:)-max(gr(1,:))/2;
gr(2,:) = gr(2,:)-max(gr(2,:))/2;

if strcmp(type,'Square')
    gr = round(gr);
    integergrid = gr(:,(abs(gr(1,:)) <= maxShift) & (abs(gr(2,:)) <= maxShift)...
        & ((mod(abs(gr(1,:)),gridSpacing)==0) & (mod(abs(gr(2,:)),gridSpacing)==0)));
else
    gr = gr*gridSpacing;
    realgrid = gr(:,abs(gr(1,:)) <= maxShift & abs(gr(2,:)) <= maxShift);
    integergrid = round(realgrid);
end

integergrid = unique(integergrid','rows')';

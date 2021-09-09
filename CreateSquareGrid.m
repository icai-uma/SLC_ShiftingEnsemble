function NeuronCoords=CreateSquareGrid(NumRows,NumCols)
% Square grid

BigNeuronCoords=zeros(2,2*NumRows,2*NumCols);

% Generate the neuron coords
for NdxRow=0:(2*NumRows-1)
    BigNeuronCoords(1,NdxRow+1,:)=NdxRow;
    for NdxCol=0:(2*NumCols-1)  
        BigNeuronCoords(2,NdxRow+1,NdxCol+1)=NdxCol;
    end
end

NeuronCoords=BigNeuronCoords(:,1:NumRows,1:NumCols);



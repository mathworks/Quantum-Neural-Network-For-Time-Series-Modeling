function circ = createPQC(inputVector, theta)
    % create parameterized quantum circuit containing composite ansatz 
    % 
    % [Reference]
    % Dimitrios Emmanoulopoulos and Sofija Dimoska. "Quantum Machine Learning in Finance:
    % Time Series Forecasting" (https://arxiv.org/abs/2202.00599) (2022) 
    %  
    % Copyright 2026 The MathWorks, Inc.

numPredictors = numel(inputVector);
[LagOrder, NumLayers] = size(theta);

if ~isequal(LagOrder, numPredictors)
    error('The row number of row of "theta" must be equal to the size of "inputVector".');
elseif mod(NumLayers,3) ~= 0
    error('The number of column of "theta" must be a multiple of 3.');
end
numCompositeAnsatz = NumLayers/3;

rxAngle = pi; % Gain of input vecotr

qB = 1 + numPredictors; % PQC size in Qubits

% create Feature Map Layer
featureMapGates = rxGate(2:qB, rxAngle.*inputVector);

% create Ansatz Layer
ansatzGates = [];
m = 0;

for n = 1:numCompositeAnsatz
    ansatzGatesXX = rxxGate(1, 2:qB, theta(:,m+1));
    ansatzGatesZZ = rzzGate(1, 2:qB, theta(:,m+2));
    ansatzGatesYY = ryyGate(1, 2:qB, theta(:,m+3));
    ansatzGates = [ansatzGates; ...
                   ansatzGatesXX; ...
                   ansatzGatesZZ; ...
                   ansatzGatesYY]; %#ok<AGROW>
    m = m + 3;
end

% create the Parameterized Quantum Circuit by concatenating 
% the Feature Map Layer and the Ansatz Layer
circ = quantumCircuit([featureMapGates; ...
        ansatzGates]);

end
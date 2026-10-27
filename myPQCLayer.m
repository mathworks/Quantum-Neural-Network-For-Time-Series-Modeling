classdef myPQCLayer < nnet.layer.Layer
    % custom layer of parameterized quantum circuit combined IsingXX, IsingYY and IsingZZ gates for time series analysis
    % 
    % [Reference]
    % 1. Zhijian Lai, Jiang Hu, Dong An, and Zaiwen Wen. "Extended Parameter Shift Rules with Minimal 
    %    Derivative Variance for Parameterized Quantum Circuits" (https://arxiv.org/abs/2508.08802) (2025)  
    % 2. Wierichs, David, Josh Izaac, Cody Wang, and Cedric Yen-Yu Lin. "General Parameter-Shift Rules 
    %    for Quantum Gradients" (https://quantum-journal.org/papers/q-2022-03-30-677/) (2022) 
    % 3. PENNYLANE, "Parameter-shift Rule" (https://pennylane.ai/qml/glossary/parameter_shift) (Modified February 25, 2026)
    %  
    % Copyright 2026 The MathWorks, Inc.

    properties      
        NumPredictors % same as window_size
    end

    properties (Learnable)
        Params % (layer.NumPredictors) x (layer.NumLayers) matrix contains learnable parameters 
    end

    properties (Constant)
        NumLayers = 6;
    end

    methods
        function layer = myPQCLayer
            % Set layer name.
            layer.Name = "PQC";

            % Set layer description.
            layer.Description = "Layer containing a parameterized quantum circuit";
        end

        function layer = initialize(layer,layout)
            % Inputs:
            %       layer  - Layer to initialize
            %       layout - Data layout, specified as a
            %       networkDataLayout object
            %                  
            %
            % Outputs:
            %       layer - Initialized layer
            % 

            % Find number of channels.
            idx = finddim(layout,"C");
            layer.NumPredictors = layout.Size(idx);

            % initialize leranable parameters based on Glorot initialization
            layer.Params = unifrnd(-sqrt(layer.NumLayers/(layer.NumPredictors+1)), ...
                                sqrt(layer.NumLayers/(layer.NumPredictors+1)), ...
                                layer.NumPredictors, layer.NumLayers);
        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %   layer - Layer to forward propagate through 
            %   X     - Input data (Batch x Channel)
            % Outputs:
            %   Z     - Output of layer forward function
            %
            
            % Z = predict(layer,X) forwards the input data X through the
            % layer and outputs the result Z at prediction time.
            Z = computeZ(X,layer);
            
        end

        function [dLdX, dLdW] = backward(layer,X,~,dLdZ,~) 

            % Backward propagate the derivative of the loss function 
            % through the function.
            %
            % Inputs:
            %   layer   - Layer to backward propagate through 
            %   X       - Layer input data (Channel x Batch)
            %   Z       - Layer output data (NA)
            %   dLdZ    - Derivative of loss with respect to layer output (NA)
            %   dLdSout - (Optional) Derivative of loss with respect 
            %             to state output (NA)
            %                            
            % Outputs:
            %   dLdX    - Derivative of loss with respect to layer input
            %   dLdW    - Derivative of loss with respect to learnable parameter
            %                    
           
            % apply parameter shift rule for the input parameters iteratively
            s = cast(pi/2, like = X); % shift parameter
            dZdX = zeros(size(X), like = X);
            for dd = 1:layer.NumPredictors
                xF = X; 
                xB = X;
                xF(dd,:) = xF(dd,:) + s;
                xB(dd,:) = xB(dd,:) - s;
                dZdX(dd,:) = 0.5.*(computeZ(xF,layer) - computeZ(xB,layer));
            end
            dLdX = bsxfun(@times, dZdX, dLdZ);

            % apply parameter shift rule for the learnable parameters iteratively
            s = cast(s, like = layer.Params); % shift parameter
            layerF = layer;
            layerB = layer;
            dLdW = zeros(layer.NumPredictors, layer.NumLayers, like = dLdZ);
            for nn = 1:layer.NumLayers
                for dd = 1:layer.NumPredictors
                    layerF.Params(dd,nn) = (layer.Params(dd,nn) + s);
                    layerB.Params(dd,nn) = (layer.Params(dd,nn) - s);
                    dZdw = 0.5.*(computeZ(X,layerF) - computeZ(X,layerB));
                    if isrow(dZdw)
                        dLdW(dd,nn) = dLdZ*transpose(dZdw);
                    else
                        dLdW(dd,nn) = transpose(dLdZ)*dZdw;
                    end
                end
            end
        end % END-of-FUNCTION (backward)
    end % END-of-METHODS
end % END-of-CLASSDEF

function Z = computeZ(X,layer)

if size(X,1) == layer.NumPredictors
    X = transpose(X);
end

batch_size = size(X,1);
obsPauli = observable(['Z', repmat('I',1,layer.NumPredictors)], 1);

Z = zeros(1,batch_size,like=X);

for ii = 1:batch_size
    circ  = createPQC(X(ii,:), layer.Params);
    Z(ii) = observe(circ, obsPauli);
end

end % END-of-FUNCTION (computeZ)
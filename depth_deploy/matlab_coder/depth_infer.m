function out = depth_infer(in) %#codegen
%DEPTH_INFER Run Depth-Anything-V2-Small inference from exported PyTorch program
%   OUT = DEPTH_INFER(IN) takes a [1, 3, 518, 784] single input and returns
%   a [1, 518, 784] single depth map output.

persistent model;
if isempty(model)
    model = loadPyTorchExportedProgram('/Users/arkadiyturevskiy/Documents/Claude/Coder_Models/Medium/Depth-Anything-V2-Small-Exported-Program.pt2');
end
out = invoke(model, in);
end

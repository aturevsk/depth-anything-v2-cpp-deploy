%% Code generation for Depth-Anything-V2-Small (no OpenMP)
outDir = fullfile(pwd, 'codegen_out');
if exist(outDir, 'dir')
    rmdir(outDir, 's');
end

cfg = coder.config('lib', 'ecoder', true);
cfg.TargetLang = 'C++';
cfg.GenerateReport = true;
cfg.GenCodeOnly = true;
cfg.EnableOpenMP = false;
cfg.SupportNonFinite = true;

dlcfg = coder.DeepLearningConfig('none');
cfg.DeepLearningConfig = dlcfg;

inputType = coder.typeof(single(0), [1 3 518 784], [false false false false]);

fprintf('Starting code generation (no OpenMP)...\n');
tic;
codegen -config cfg depth_infer -args {inputType} -d codegen_out -report
t = toc;
fprintf('Code generation completed in %.1f seconds.\n', t);

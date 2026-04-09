# NNCPE

## Instructions 

To produce an output TTree with this EDAnalyzer, run the following steps:

```
export SCRAM_ARCH=el9_amd64_gcc12
cmsrel CMSSW_16_1_0_pre4 
cd CMSSW_16_1_0_pre4/
cd src/
cmsenv
git clone https://github.com/ammitra/NNCPE.git
scram b -j 4
cd NNCPE/ExtractCPEInfo/python
cmsRun ConfFile_cfg.py
```

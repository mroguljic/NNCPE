# coding: utf-8

import os
import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

from Configuration.Eras.Era_Run3_cff import Run3

n_events = 5





# define the process to run
#process = cms.Process('REC',Run2_2018)
process = cms.Process('REC',Run3)
#pf_badHcalMitigation)
# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Demo.ExtractCPEInfo.extractCPEInfo_cfi')
#process.load('RecoLocalTracker.SiPixelRecHits.plugins.extractCPEInfo_cfi')

#process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')

# to get the conditions you need a GT
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# MC
#process.GlobalTag = GlobalTag(process.GlobalTag, '105X_upgrade2018_design_v3', '') #phase-1 2018 unirradiated
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_realistic', '')
# data
#process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun2_v7', '')
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
#process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
#process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
#process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
# force Generic reco
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(n_events))
process.source = cms.Source("PoolSource",
  # data
  #fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/data/Run2018C/SingleMuon/RAW/v1/000/320/040/00000/407FB3FD-A78E-E811-B816-FA163E120D15.root")
  # MC

fileNames=cms.untracked.vstring("file:step2_DIGI_L1_DIGI2RAW_HLT.root"),
#fileNames=cms.untracked.vstring("file:TTbar_13TeV_TuneCUETP8M1_cfi_MC_500_phase1_2018_realistic.root"),
#eventsToSkip=cms.untracked.VEventRange('1:8-1:10','1:67-1:69','1:388-1:390'),
eventsToSkip=cms.untracked.VEventRange('1:94-1:96','1:173-1:175'),
# data
  #fileNames=cms.untracked.vstring("file:52A3B4C3-328E-E811-85D6-FA163E3AB92A.root")
#skipEvents=cms.untracked.uint32(385)

)
print("Using global tag "+process.GlobalTag.globaltag._value)

# process options
process.options = cms.untracked.PSet(allowUnscheduled=cms.untracked.bool(True),wantSummary=cms.untracked.bool(True))




process.ExtractCPEInfo = cms.EDAnalyzer('ExtractCPEInfo',
 fname = cms.string("test.root"),
 associateRecoTracks = cms.bool(False),
 associateStrip = cms.bool(False),
 associatePixel = cms.bool(True),
 pixelSimLinkSrc = cms.string("simSiPixelDigis"),
 stripSimLinkSrc = cms.string("simSiStripDigis"),
 ROUList = cms.vstring(
    'TrackerHitsPixelBarrelLowTof', 
    'TrackerHitsPixelBarrelHighTof', 
    'TrackerHitsPixelEndcapLowTof', 
 )
)
# define what to run in the path
process.raw2digi_step = cms.Path(process.RawToDigi)   
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.siPixelClusters_step = process.siPixelClusters

process.ExtractCPEInfo_step = cms.Path(process.ExtractCPEInfo)

# potentially for the det angle approach
#process.schedule = cms.Schedule(
#  process.raw2digi_step,
#  process.siPixelClusters_step,
#  process.pixelCPECNN_step
#)

# for the track angle approach
process.schedule = cms.Schedule(
  process.raw2digi_step,
  process.reconstruction_step,
  process.ExtractCPEInfo_step,
  process.endjob_step
  )


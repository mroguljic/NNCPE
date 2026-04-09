/*
 * Example plugin to demonstrate the direct multi-threaded inference on CNN with TensorFlow 2.
 */

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/time.h>

#include <TF1.h>
#include "Math/MinimizerOptions.h"
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TObject.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "Math/DistFunc.h"
#include "TTree.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
//#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"  
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"


#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class TObject;
class TTree;
class TH1D;
class TFile;

using namespace std;
using namespace edm;
using namespace reco;
// define the cache object
struct CacheData {
    CacheData()  {}
};

class ExtractCPEInfo : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
    explicit ExtractCPEInfo(const edm::ParameterSet&, const CacheData*);
    ~ExtractCPEInfo(){};

    static void fillDescriptions(edm::ConfigurationDescriptions&);
    // two additional static methods for handling the global cache
    static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
    static void globalEndJob(const CacheData*); // does it have to be static
private:
    void beginJob();
    void analyze(const edm::Event&, const edm::EventSetup&);
    void endRun(edm::Run const&, edm::EventSetup const&) override;
    void endJob();
    //void endRun();
    void ResetVars();

    TFile *out_File; 
    TTree *out_Tree;

    std::string fname;

    static const int MAXCLUSTER = 80000;
    static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit

    int Track_idx;

    // BPIX
    int Layer;        // increasing r
    int Ladder;       // increasing phi
    int Module;       // increasing z (this value is always 1 for FPIX, so only track it for BPIX)

    // FPIX
    int Side;         // FPIX- or FPIX+
    int Disk;         // increasing abs(z)
    int Blade;        // increasing phi and r
    int Panel;        // fwd/backward

    float CotAlpha;
    float CotBeta;
    float CotAlpha_detAngle;
    float CotBeta_detAngle;
    float Cluster_raw[TXSIZE][TYSIZE];
    float Cluster_xRaw[TXSIZE];
    float Cluster_yRaw[TYSIZE];
    float Cluster[TXSIZE][TYSIZE];
    float Cluster_x[TXSIZE];
    float Cluster_y[TYSIZE];
    int Tx_size = TXSIZE;
    int Ty_size = TYSIZE;

    int nSimHit;
    float SimHit_x[SIMHITPERCLMAX];    // X local position of simhit
    float SimHit_y[SIMHITPERCLMAX];

    float Generic_x;
    float Generic_y;
    float Generic_xError;
    float Generic_yError;
    float Generic_dx[SIMHITPERCLMAX];
    float Generic_dxPull[SIMHITPERCLMAX];
    float Generic_dy[SIMHITPERCLMAX];
    float Generic_dyPull[SIMHITPERCLMAX];

    edm::InputTag fTrackCollectionLabel;
    int count=0;

    edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> TrackerTopoToken;
    TrackerHitAssociator::Config trackerHitAssociatorConfig_;
};

std::unique_ptr<CacheData> ExtractCPEInfo::initializeGlobalCache(const edm::ParameterSet& config)
{
    CacheData* cacheData = new CacheData();
    return std::unique_ptr<CacheData>(cacheData);
}

void ExtractCPEInfo::globalEndJob(const CacheData* cacheData) {
    printf("in global end job\n");

}
void ExtractCPEInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // defining this function will lead to a *_cfi file being generated when compiling
    edm::ParameterSetDescription desc;
    desc.add<std::string>("fname");
    desc.add<bool>("associateRecoTracks");
    desc.add<std::string>("pixelSimLinkSrc");
    desc.add<std::string>("stripSimLinkSrc");
    desc.add<std::vector<std::string>>("ROUList");
    desc.add<bool>("associatePixel");
    desc.add<bool>("associateStrip");
    descriptions.addWithDefaultLabel(desc);
}

void ExtractCPEInfo::ResetVars(){
    Ladder = 0;       // increasing phi
    Module = 0;       // increasing z (this value is always 1 for FPIX, so only track it for BPIX)
    Side   = 0;       // FPIX- or FPIX+
    Disk   = 0;       // increasing abs(z)
    Blade  = 0;       // increasing phi and r
    Panel  = 0;       // fwd/backward

    Layer = 0;
    Track_idx = 0;
    CotAlpha = 0.f;
    CotBeta = 0.f;
    CotAlpha_detAngle = 0.f;
    CotBeta_detAngle = 0.f;
    nSimHit = 0;
    Generic_x = 0.f;
    Generic_y = 0.f;
    Generic_xError = 0.f;
    Generic_yError = 0.f;
    for(int i = 0; i < TXSIZE; i++){
        Cluster_xRaw[i] = 0.f;
        Cluster_x[i] = 0.f;
        for(int j = 0; j < TYSIZE; j++){
            Cluster_raw[i][j] = 0.f;
            Cluster[i][j] = 0.f;
        }
    }
    for(int j = 0; j < TYSIZE; j++){
        Cluster_yRaw[j] = 0.f;
        Cluster_y[j] = 0.f;
    }

    for (int i = 0; i < SIMHITPERCLMAX; i++){
        SimHit_x[i] = std::numeric_limits<float>::max();
        SimHit_y[i] = std::numeric_limits<float>::max();
        Generic_dx[i] = std::numeric_limits<float>::max();
        Generic_dxPull[i] = std::numeric_limits<float>::max();
        Generic_dy[i] = std::numeric_limits<float>::max();
        Generic_dyPull[i] = std::numeric_limits<float>::max();
    }

}

ExtractCPEInfo::ExtractCPEInfo(const edm::ParameterSet& config, const CacheData* cacheData)
:fname(config.getParameter<std::string>("fname")),fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
trackerHitAssociatorConfig_(config, consumesCollector()) {
    TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
    TrackerTopoToken        = esConsumes <TrackerTopology, TrackerTopologyRcd>();
    ResetVars();

    count = 0;
    out_File = new TFile(fname.c_str(), "RECREATE");
    out_Tree = new TTree("Events", "Tree For NN Training");
    out_Tree->Branch("Layer", &Layer, "Layer/I");
    out_Tree->Branch("Track_idx", &Track_idx, "Track_idx/I");
    out_Tree->Branch("Ladder", &Ladder, "Ladder/I");
    out_Tree->Branch("Module", &Module, "Module/I");
    out_Tree->Branch("Side", &Side, "Side/I");
    out_Tree->Branch("Disk", &Disk, "Disk/I");
    out_Tree->Branch("Blade", &Blade, "Blade/I");
    out_Tree->Branch("Panel", &Panel, "Panel/I");
    out_Tree->Branch("CotAlpha",&CotAlpha, "CotAlpha/F");
    out_Tree->Branch("CotBeta",&CotBeta, "CotBeta/F");

    out_Tree->Branch("CotAlpha_detAngle", &CotAlpha_detAngle, "CotAlpha_detAngle/F");
    out_Tree->Branch("CotBeta_detAngle", &CotBeta_detAngle, "CotBeta_detAngle/F");
    // ROOT does not support variable length 2D arrays so this is hardcoded instead of using TXSIZE (13) and TYSIZE (21) -- is there a better way?
    out_Tree->Branch("Cluster", &Cluster, "Cluster[13][21]/F");
    out_Tree->Branch("Cluster_x", &Cluster_x, "Cluster_x[13]/F");
    out_Tree->Branch("Cluster_y", &Cluster_y, "Cluster_y[21]/F");
    out_Tree->Branch("Cluster_raw", &Cluster_raw, "Cluster_raw[13][21]/F");
    out_Tree->Branch("Cluster_xRaw", &Cluster_xRaw, "Cluster_xRaw[13]/F");
    out_Tree->Branch("Cluster_yRaw", &Cluster_yRaw, "Cluster_yRaw[21]/F");

    out_Tree->Branch("Generic_x", &Generic_x, "Generic_x/F");
    out_Tree->Branch("Generic_y", &Generic_y, "Generic_y/F");
    out_Tree->Branch("Generic_xError", &Generic_xError, "Generic_xError/F");
    out_Tree->Branch("Generic_yError", &Generic_yError, "Generic_yError/F");

    out_Tree->Branch("nSimHit", &nSimHit, "nSimHit/I");
    out_Tree->Branch("SimHit_x", &SimHit_x, "SimHit_x[nSimHit]/F");
    out_Tree->Branch("SimHit_y", &SimHit_y, "SimHit_y[nSimHit]/F");
    out_Tree->Branch("Generic_dx", &Generic_dx, "Generic_dx[nSimHit]/F");
    out_Tree->Branch("Generic_dy", &Generic_dy, "Generic_dy[nSimHit]/F");
    out_Tree->Branch("Generic_dxPull", &Generic_dxPull, "Generic_dxPull[nSimHit]/F");
    out_Tree->Branch("Generic_dyPull", &Generic_dyPull, "Generic_dyPull[nSimHit]/F");
}


void ExtractCPEInfo::beginJob() {

}

void ExtractCPEInfo::endJob() {
    printf("in end job\n");

}
void ExtractCPEInfo::endRun(edm::Run const&, edm::EventSetup const&) {
    printf("in end run\n");
    out_File->cd();
    out_Tree->Write();
    out_File->Close();

}





void ExtractCPEInfo::analyze(const edm::Event& event, const edm::EventSetup& setup) {
    std::cout<<"TEST"<<std::endl;
    edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(TrackerTopoToken);
    auto const& tkTpl = *tTopoHandle;

    //get the map
    edm::Handle<reco::TrackCollection> tracks;

    try {
        event.getByToken(TrackToken, tracks);
    }catch (cms::Exception &ex) {
        cout << "No Track collection with label " << fTrackCollectionLabel << endl;
    }
    if (! tracks.isValid()) {
        cout << "track collection is not valid" <<endl;
        return;
    }

    float clusbuf_temp[TXSIZE][TYSIZE];
    int idx = -1;
    for (auto const& track : *tracks) {
        idx++;
        auto const& trajParams = track.extra()->trajParams();
        assert(trajParams.size() == track.recHitsSize());
        auto hb = track.recHitsBegin();
        for (unsigned int h = 0; h < track.recHitsSize(); h++) {
            ResetVars();
            Track_idx = idx;
            auto hit = *(hb + h);
            if (!hit->isValid())
                continue;
            if (hit->geographicalId().det() != DetId::Tracker) {
                continue;
            }
            auto id = hit->geographicalId();
            DetId hit_detId = hit->geographicalId();

            // check that we are in the pixel detector
            auto subdetid = (id.subdetId());

            if (subdetid == PixelSubdetector::PixelBarrel) { //&& subdetid != PixelSubdetector::PixelEndcap)
                Layer  = tkTpl.pxbLayer(hit_detId);
                Ladder = tkTpl.pxbLadder(hit_detId);
                Module = tkTpl.pxbModule(hit_detId);
                Side   = -1;
                Disk   = -1;
                Blade  = -1;
                Panel  = -1;
            }
            else if (subdetid == PixelSubdetector::PixelEndcap) {
                Layer  = -1;
                Ladder = -1;
                Module = -1;
                Side   = tkTpl.pxfSide(hit_detId);
                Disk   = tkTpl.pxfDisk(hit_detId);
                Blade  = tkTpl.pxfBlade(hit_detId);
                Panel  = tkTpl.pxfPanel(hit_detId);
            }
            else {
                continue;
            }

            //Layer = tkTpl.pxbLayer(hit_detId);

            auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
            if (!pixhit)
                continue;

            //some initialization
            for(int j=0; j<TXSIZE; ++j) {
                for(int i=0; i<TYSIZE; ++i) {
                    clusbuf_temp[j][i] = 0.;
                }
            }

            // get the cluster
            auto clustp = pixhit->cluster();
            if (clustp.isNull())
                continue;
            auto const& cluster = *clustp;
            int row_offset = cluster.minPixelRow();
            int col_offset = cluster.minPixelCol();
            // CALCULATING ANGLE
            auto const& ltp = trajParams[h];

            CotAlpha = ltp.dxdz();
            CotBeta = ltp.dydz();
            //https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEBase.cc#L263-L272
            LocalPoint trk_lp = ltp.position();
            float trk_lp_x = trk_lp.x();
            float trk_lp_y = trk_lp.y();

            auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
            if(!geomdetunit) continue;
            auto const& topol = geomdetunit->specificTopology();

            auto const& theOrigin = geomdetunit->surface().toLocal(GlobalPoint(0, 0, 0));
            LocalPoint lp2 = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
            auto gvx = lp2.x() - theOrigin.x();
            auto gvy = lp2.y() - theOrigin.y();
            auto gvz = -1.f /theOrigin.z();
            // calculate angles
            CotAlpha_detAngle = gvx * gvz;
            CotBeta_detAngle = gvy * gvz;

            // LOADING CLUSTER
            int mrow = 0, mcol = 0;
            for (int i = 0; i != cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x);
                int icol = int(pix.y);
                mrow = std::max(mrow, irow);
                mcol = std::max(mcol, icol);
            }
            mrow -= row_offset;
            mrow += 1;
            mrow = std::min(mrow, TXSIZE);
            mcol -= col_offset;
            mcol += 1;
            mcol = std::min(mcol, TYSIZE);
            assert(mrow > 0);
            assert(mcol > 0);

            int n_double_x = 0, n_double_y = 0;
            int clustersize = 0;
            int double_row[5], double_col[5];
            for(int i=0;i<5;i++){
                double_row[i]=-1;
                double_col[i]=-1;
            }

            int irow_sum = 0, icol_sum = 0;
            for (int i = 0; i < cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x) - row_offset;
                int icol = int(pix.y) - col_offset;
                if ((irow >= mrow) || (icol >= mcol)) continue;
                if ((int)pix.x == 79 || (int)pix.x == 80){
                    int flag=0;
                    for(int j=0;j<5;j++){
                        if(irow==double_row[j]) {flag = 1; break;}
                    }
                    if(flag!=1) {double_row[n_double_x]=irow; n_double_x++;}
                }
                if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51){
                    int flag=0;
                    for(int j=0;j<5;j++){
                        if(icol==double_col[j]) {flag = 1; break;}
                    }
                    if(flag!=1) {double_col[n_double_y]=icol; n_double_y++;}
                }
                irow_sum+=irow;
                icol_sum+=icol;
                clustersize++;
            }

            if(clustersize==0){printf("EMPTY CLUSTER, SKIPPING\n");continue;}
            if(n_double_x>2 or n_double_y>2){
                continue; //currently can only deal with single double pix
            }
            n_double_x=0; n_double_y=0;
            int clustersize_x = cluster.sizeX(), clustersize_y = cluster.sizeY();
            int mid_x = round(float(irow_sum)/float(clustersize));
            int mid_y = round(float(icol_sum)/float(clustersize));
            int offset_x = 6 - mid_x;
            int offset_y = 10 - mid_y;
            for (int i = 0; i < cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x) - row_offset + offset_x;
                int icol = int(pix.y) - col_offset + offset_y;

                if ((irow >= mrow+offset_x) || (icol >= mcol+offset_y)){
                    printf("irow or icol exceeded, SKIPPING. irow = %i, mrow = %i, offset_x = %i,icol = %i, mcol = %i, offset_y = %i\n",irow,mrow,offset_x,icol,mcol,offset_y);
                    continue;
                }
                if ((int)pix.x == 79 || (int)pix.x == 80){
                    int flag=0;
                    for(int j=0;j<5;j++){
                        if(irow==double_row[j]) {flag = 1; break;}
                    }
                    if(flag!=1) {double_row[n_double_x]=irow; n_double_x++;}
                }
                if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
                    int flag=0;
                    for(int j=0;j<5;j++){
                        if(icol==double_col[j]) {flag = 1; break;}
                    }
                    if(flag!=1) {double_col[n_double_y]=icol; n_double_y++;}
                }
                clusbuf_temp[irow][icol] = float(pix.adc)/25000.;
            }

            if(n_double_x==1 && clustersize_x>12) {printf("clustersize_x > 12, SKIPPING\n"); continue;} // NEED TO FIX CLUSTERSIZE COMPUTATION
            if(n_double_x==2 && clustersize_x>11) {printf("clustersize_x > 11, SKIPPING\n"); continue;}
            if(n_double_y==1 && clustersize_y>20) {printf("clustersize_y = %i > 20, SKIPPING\n", clustersize_y);continue;}
            if(n_double_y==2 && clustersize_x>19) {printf("clustersize_y = %i > 19, SKIPPING\n", clustersize_y);continue;}
            //first deal with double width pixels in x
            int k=0,m=0;
            for(int i=0;i<TXSIZE;i++){
                if(i==double_row[m] and clustersize_x>1){
                    printf("TREATING DPIX%i IN X\n",m+1);
                    for(int j=0;j<TYSIZE;j++){
                        Cluster_raw[i][j]=clusbuf_temp[k][j]/2.;
                        Cluster_raw[i+1][j]=clusbuf_temp[k][j]/2.;
                    }
                    i++;
                    if(m==0 and n_double_x==2) {
                        double_row[1]++;
                        m++;
                    }
                }
                else{
                    for(int j=0;j<TYSIZE;j++){
                        Cluster_raw[i][j]=clusbuf_temp[k][j];
                    }
                }
                k++;
            }
            k=0;m=0;
            for(int i=0;i<TXSIZE;i++){
                for(int j=0;j<TYSIZE;j++){
                    clusbuf_temp[i][j]=Cluster_raw[i][j];
                    Cluster_raw[i][j]=0.;
                }
            }
            for(int j=0;j<TYSIZE;j++){
                if(j==double_col[m] and clustersize_y>1){
                    printf("TREATING DPIX%i IN Y\n",m+1);
                    for(int i=0;i<TXSIZE;i++){
                        Cluster_raw[i][j]=clusbuf_temp[i][k]/2.;
                        Cluster_raw[i][j+1]=clusbuf_temp[i][k]/2.;
                    }
                    j++;
                    if(m==0 and n_double_y==2) {
                        double_col[1]++;
                        m++;
                    }
                }
                else{
                    for(int i=0;i<TXSIZE;i++){
                        Cluster_raw[i][j]=clusbuf_temp[i][k];
                    }
                }
                k++;
            }
            //compute the 1d projection & compute cluster max
            float cluster_max = 0., cluster_max_2d = 0.;
            for(int i = 0;i < TXSIZE; i++){
                for(int j = 0; j < TYSIZE; j++){
                    Cluster_xRaw[i] += Cluster_raw[i][j];
                    Cluster_yRaw[j] += Cluster_raw[i][j];
                    if(Cluster_raw[i][j]>cluster_max_2d) cluster_max_2d = Cluster_raw[i][j];
                }
                if(Cluster_xRaw[i] > cluster_max) cluster_max = Cluster_xRaw[i] ;
            }
            for(int j = 0; j < TYSIZE; j++){
                if(Cluster_yRaw[j] > cluster_max) cluster_max = Cluster_yRaw[j] ;
            }

            //normalize 2d inputs
            for(int i = 0;i < TXSIZE; i++){
                for (int j = 0; j < TYSIZE; j++)
                {
                    Cluster[i][j] = Cluster_raw[i][j]/cluster_max_2d;
                }
                Cluster_x[i] = Cluster_xRaw[i]/cluster_max;
            }
            for(int j = 0; j < TYSIZE; j++){
                Cluster_y[j] = Cluster_yRaw[j]/cluster_max;
            }

            // get the generic position
            Generic_x = pixhit->localPosition().x();
            Generic_xError = TMath::Sqrt(pixhit->localPositionError().xx());
            Generic_y = pixhit->localPosition().y();
            Generic_yError = TMath::Sqrt(pixhit->localPositionError().yy());
            //get sim hits

            std::vector<PSimHit> vec_simhits_assoc;
            TrackerHitAssociator *associate(0);

            associate = new TrackerHitAssociator(event,trackerHitAssociatorConfig_);
            vec_simhits_assoc.clear();
            vec_simhits_assoc = associate->associateHit(*pixhit);

            int iSimHit = 0;

            for (std::vector<PSimHit>::const_iterator m = vec_simhits_assoc.begin();
                m < vec_simhits_assoc.end() && iSimHit < SIMHITPERCLMAX; ++m)
            {
                SimHit_x[iSimHit]    = ( m->entryPoint().x() + m->exitPoint().x() ) / 2.0;
                SimHit_y[iSimHit]    = ( m->entryPoint().y() + m->exitPoint().y() ) / 2.0;

                ++iSimHit;

            } // end sim hit loop
            nSimHit = iSimHit;
            std::cout<<nSimHit<<std::endl;
            for(int i = 0; i < 10; i++)
                std::cout<<"SimHit_x "<<SimHit_x[i]<<std::endl;
            for(int i = 0;i<nSimHit;i++){
                Generic_dx[i] = Generic_x - SimHit_x[i];
                Generic_dy[i] = Generic_y - SimHit_y[i];
                Generic_dxPull[i] = Generic_dx[i]/Generic_xError;
                Generic_dyPull[i] = Generic_dy[i]/Generic_yError;
            }
            count++;
            out_Tree->Fill();
        }
    }

    printf("total cluster count = %i\n",count);

}
DEFINE_FWK_MODULE(ExtractCPEInfo);

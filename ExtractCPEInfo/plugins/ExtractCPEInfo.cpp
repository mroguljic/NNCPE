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
#include <limits>
#include <tuple>

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
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

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
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
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
    static bool isWideRow(int absRow);
    static bool isWideCol(int absCol);
    static bool isWidePixel(int absRow, int absCol);

    TFile *out_File; 
    TTree *out_Tree;

    std::string fname;

    static const int SIMHITPERCLMAX = 10; // max number of simhits associated with a cluster/rechit
    static constexpr float CHARGENORM = 25000.f; // value to divide cluster charge by
    int Track_idx;

    // BPIX
    int Layer;        // increasing r
    int Ladder;       // increasing phi
    int Module;       // increasing z (this value is always 1 for FPIX)

    // FPIX
    int Side;         // FPIX- or FPIX+
    int Disk;         // increasing abs(z)
    int Blade;        // increasing phi and r
    int Panel;        // fwd/backward

    float CotAlpha;
    float CotBeta;
    float CotAlpha_detAngle;
    float CotBeta_detAngle;
    float ClusterCenter_x;
    float ClusterCenter_y;
    int Row_offset;
    int Col_offset;
    int Cluster_size;
    int Cluster_sizeX;
    int Cluster_sizeY;

    float Cluster_charge; // total cluster charge, in units of CHARGENORM electrons
    //"Raw" charge is units of CHARGENORM electrons
    float Cluster_raw[TXSIZE][TYSIZE]; // TXSIZE and TYSIZE are cluster matrix dimensions (defined for template reco, we use the same dimensions)
    float Cluster_xRaw[TXSIZE];
    float Cluster_yRaw[TYSIZE];
    //Normalized so that the max pixel charge is 1 in both the 2D ckluster and the 1D projections
    float Cluster[TXSIZE][TYSIZE]; // Normalized so that the max pixel charge in the cluster is 1
    float Cluster_x[TXSIZE];
    float Cluster_y[TYSIZE];

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

    float Template_x;
    float Template_y;
    float Template_xError;
    float Template_yError;
    float Template_dx[SIMHITPERCLMAX];
    float Template_dxPull[SIMHITPERCLMAX];
    float Template_dy[SIMHITPERCLMAX];
    float Template_dyPull[SIMHITPERCLMAX];

    edm::InputTag fTrackCollectionLabel;
    int count=0;

    bool useGenericCPE_;
    bool useTemplateCPE_;
    edm::ESInputTag genericCPELabel_;
    edm::ESInputTag templateCPELabel_;

    edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> TrackerTopoToken;
    edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> GenericCPEToken;
    edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> TemplateCPEToken;
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
    desc.add<bool>("useGenericCPE", true);
    desc.add<bool>("useTemplateCPE", false);
    desc.add<edm::ESInputTag>("genericCPE", edm::ESInputTag("", "PixelCPEGeneric"));
    desc.add<edm::ESInputTag>("templateCPE", edm::ESInputTag("", "PixelCPEClusterRepair"));
    descriptions.addWithDefaultLabel(desc);
}

bool ExtractCPEInfo::isWideRow(int absRow) {
    return absRow == 79 || absRow == 80;
}

bool ExtractCPEInfo::isWideCol(int absCol) {
    return (absCol % 52 == 0) || (absCol % 52 == 51);
}

bool ExtractCPEInfo::isWidePixel(int absRow, int absCol) {
    return isWideRow(absRow) || isWideCol(absCol);
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
    Cluster_size = 0;
    Cluster_sizeX = 0;
    Cluster_sizeY = 0;
    Row_offset = -1;
    Col_offset = -1;
    ClusterCenter_x = 0.f;
    ClusterCenter_y = 0.f;
    Cluster_charge = 0.f;
    nSimHit = 0;
    Generic_x = 0.f;
    Generic_y = 0.f;
    Generic_xError = 0.f;
    Generic_yError = 0.f;
    Template_x = 0.f;
    Template_y = 0.f;
    Template_xError = 0.f;
    Template_yError = 0.f;

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
        Template_dx[i] = std::numeric_limits<float>::max();
        Template_dxPull[i] = std::numeric_limits<float>::max();
        Template_dy[i] = std::numeric_limits<float>::max();
        Template_dyPull[i] = std::numeric_limits<float>::max();
    }

}

ExtractCPEInfo::ExtractCPEInfo(const edm::ParameterSet& config, const CacheData* cacheData)
:fname(config.getParameter<std::string>("fname")),fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
trackerHitAssociatorConfig_(config, consumesCollector()) {
    useGenericCPE_ = config.getParameter<bool>("useGenericCPE");
    useTemplateCPE_ = config.getParameter<bool>("useTemplateCPE");
    genericCPELabel_ = config.getParameter<edm::ESInputTag>("genericCPE");
    templateCPELabel_ = config.getParameter<edm::ESInputTag>("templateCPE");

    if (!useGenericCPE_ && !useTemplateCPE_) {
        throw cms::Exception("Configuration")
            << "ExtractCPEInfo requires at least one CPE enabled. "
            << "Set useGenericCPE=True and/or useTemplateCPE=True.";
    }

    TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
    TrackerTopoToken        = esConsumes <TrackerTopology, TrackerTopologyRcd>();
    if (useGenericCPE_) {
        GenericCPEToken = esConsumes(genericCPELabel_);
    }
    if (useTemplateCPE_) {
        TemplateCPEToken = esConsumes(templateCPELabel_);
    }
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
    out_Tree->Branch("Row_offset", &Row_offset, "Row_offset/I");
    out_Tree->Branch("Col_offset", &Col_offset, "Col_offset/I");
    out_Tree->Branch("ClusterCenter_x", &ClusterCenter_x, "ClusterCenter_x/F");
    out_Tree->Branch("ClusterCenter_y", &ClusterCenter_y, "ClusterCenter_y/F");
    out_Tree->Branch("Cluster_charge", &Cluster_charge, "Cluster_charge/F");


    out_Tree->Branch("CotAlpha_detAngle", &CotAlpha_detAngle, "CotAlpha_detAngle/F");
    out_Tree->Branch("CotBeta_detAngle", &CotBeta_detAngle, "CotBeta_detAngle/F");
    out_Tree->Branch("Cluster_size", &Cluster_size, "Cluster_size/I");
    out_Tree->Branch("Cluster_sizeX", &Cluster_sizeX, "Cluster_sizeX/I");
    out_Tree->Branch("Cluster_sizeY", &Cluster_sizeY, "Cluster_sizeY/I");
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

    out_Tree->Branch("Template_x", &Template_x, "Template_x/F");
    out_Tree->Branch("Template_y", &Template_y, "Template_y/F");
    out_Tree->Branch("Template_xError", &Template_xError, "Template_xError/F");
    out_Tree->Branch("Template_yError", &Template_yError, "Template_yError/F");

    out_Tree->Branch("nSimHit", &nSimHit, "nSimHit/I");
    out_Tree->Branch("SimHit_x", &SimHit_x, "SimHit_x[nSimHit]/F");
    out_Tree->Branch("SimHit_y", &SimHit_y, "SimHit_y[nSimHit]/F");
    out_Tree->Branch("Generic_dx", &Generic_dx, "Generic_dx[nSimHit]/F");
    out_Tree->Branch("Generic_dy", &Generic_dy, "Generic_dy[nSimHit]/F");
    out_Tree->Branch("Generic_dxPull", &Generic_dxPull, "Generic_dxPull[nSimHit]/F");
    out_Tree->Branch("Generic_dyPull", &Generic_dyPull, "Generic_dyPull[nSimHit]/F");
    out_Tree->Branch("Template_dx", &Template_dx, "Template_dx[nSimHit]/F");
    out_Tree->Branch("Template_dy", &Template_dy, "Template_dy[nSimHit]/F");
    out_Tree->Branch("Template_dxPull", &Template_dxPull, "Template_dxPull[nSimHit]/F");
    out_Tree->Branch("Template_dyPull", &Template_dyPull, "Template_dyPull[nSimHit]/F");
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
    edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(TrackerTopoToken);
    auto const& tkTpl = *tTopoHandle;

    edm::ESHandle<PixelClusterParameterEstimator> genericCPEHandle;
    edm::ESHandle<PixelClusterParameterEstimator> templateCPEHandle;
    const PixelClusterParameterEstimator* genericCPE = nullptr;
    const PixelClusterParameterEstimator* templateCPE = nullptr;
    if (useGenericCPE_) {
        genericCPEHandle = setup.getHandle(GenericCPEToken);
        if (!genericCPEHandle.isValid()) {
            cout << "generic PixelClusterParameterEstimator is not valid" << endl;
            return;
        }
        genericCPE = genericCPEHandle.product();
    }
    if (useTemplateCPE_) {
        templateCPEHandle = setup.getHandle(TemplateCPEToken);
        if (!templateCPEHandle.isValid()) {
            cout << "template PixelClusterParameterEstimator is not valid" << endl;
            return;
        }
        templateCPE = templateCPEHandle.product();
    }

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

    float clusbuf_temp[TXSIZE][TYSIZE]; // Holds cluster charge before unpacking wide pixels
    int idx = -1;
    int double_pixel_buffer_size = 5; // we don't expect more than 2 double pixels in either direction, 5 for safety margin
    int double_row[double_pixel_buffer_size], double_col[double_pixel_buffer_size];

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

            auto subdetid = (id.subdetId());

            if (subdetid == PixelSubdetector::PixelBarrel) {
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

            auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
            if (!pixhit)
                continue;

            for(int i=0; i<TXSIZE; ++i) {
                for(int j=0; j<TYSIZE; ++j) {
                    clusbuf_temp[i][j] = 0.;
                }
            }

            // get the cluster
            auto clustp = pixhit->cluster();
            if (clustp.isNull())
                continue;
            auto const& cluster = *clustp;
            Row_offset = cluster.minPixelRow();
            Col_offset = cluster.minPixelCol();
            int row_offset = Row_offset;
            int col_offset = Col_offset;
            auto const& ltp = trajParams[h];
            CotAlpha = ltp.dxdz();
            CotBeta = ltp.dydz();
            auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
            if(!geomdetunit) continue;
            auto const& topol = geomdetunit->specificTopology();

            auto const& theOrigin = geomdetunit->surface().toLocal(GlobalPoint(0, 0, 0));
            LocalPoint lp2 = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y())); // cluster.x() and .y() are the cluster charge barycenters
            auto gvx = lp2.x() - theOrigin.x();
            auto gvy = lp2.y() - theOrigin.y();
            auto gvz = -1.f /theOrigin.z();
            CotAlpha_detAngle = gvx * gvz;
            CotBeta_detAngle = gvy * gvz;

            // LOADING CLUSTER
            int mrow = 0, mcol = 0; //maximum row and column of the cluster
            for (int i = 0; i != cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x);
                int icol = int(pix.y);
                mrow = std::max(mrow, irow);
                mcol = std::max(mcol, icol);
            }
            mrow -= row_offset; // convert to local cluster coordinates, offset is the min row/col of the cluster
            mrow += 1;
            mrow = std::min(mrow, TXSIZE); // limit cluster size to the dimensions of the input matrix for the NN / template reco
            mcol -= col_offset;
            mcol += 1;
            mcol = std::min(mcol, TYSIZE);
            assert(mrow > 0);
            assert(mcol > 0);

            int n_double_x = 0, n_double_y = 0;
            int clustersize = 0; // number of pixels in the cluster, wide pixels still count as 1 here!

            for(int i=0;i<double_pixel_buffer_size;i++){
                double_row[i]=-1;
                double_col[i]=-1;
            }

            int irow_sum = 0, icol_sum = 0;
            for (int i = 0; i < cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x) - row_offset;
                int icol = int(pix.y) - col_offset;
                int absRow = int(pix.x);
                int absCol = int(pix.y);
                if ((irow >= mrow) || (icol >= mcol)) continue;
                if (isWidePixel(absRow, absCol)) {
                    if (isWideRow(absRow)) { // Phase-1 specific wide-pixel rows
                        int flag=0; // check if this row is already counter as a double pixel row
                        for(int j=0;j<n_double_x;j++){
                            if(irow==double_row[j]) {flag = 1; break;}
                        }
                        if(flag!=1) {double_row[n_double_x]=irow; n_double_x++;}
                    }
                    if (isWideCol(absCol)) { // Phase-1 specific wide-pixel columns
                        int flag=0;  // check if this column is already counter as a double pixel column
                        for(int j=0;j<n_double_y;j++){
                            if(icol==double_col[j]) {flag = 1; break;}
                        }
                        if(flag!=1) {double_col[n_double_y]=icol; n_double_y++;}
                    }
                }
                irow_sum+=irow;
                icol_sum+=icol;
                clustersize++;
            }

            if(clustersize==0){printf("EMPTY CLUSTER, SKIPPING\n");continue;}
            if(n_double_x>2 or n_double_y>2){
                continue; //currently only dealing with up to 2 double pixels in either direction. Covers most (all?) cases
            }

            Cluster_size = cluster.size(); // this is the total number of pixels in the cluster, wide pixels still count as 1
            Cluster_sizeX = cluster.sizeX() + n_double_x; // if there is a double pixel in x/y, the effective cluster size in x/y is increased by 1, we will unpack wide pixels later and fill the input matrix accordingly
            Cluster_sizeY = cluster.sizeY() + n_double_y;
            int mid_x = round(float(irow_sum)/float(clustersize)); //pixel that will be flagged as the center in the input matrix
            int mid_y = round(float(icol_sum)/float(clustersize));

            float pitch_center_x = 0.5f;
            float pitch_center_y = 0.5f;
            // if the center pixel is wide, it gets split into two, so the center of the selected mid-pix will actually be at the quarter of the origina, wide pixel's pitch
            if (isWideRow(mid_x+row_offset)) pitch_center_x = 0.25f; 
            if (isWideCol(mid_y+col_offset)) pitch_center_y = 0.25f;
            MeasurementPoint meas_center_pix(row_offset + mid_x + pitch_center_x, col_offset + mid_y + pitch_center_y); // lower-left corner
            LocalPoint local_center_pix = topol.localPosition(meas_center_pix);
            ClusterCenter_x = local_center_pix.x();
            ClusterCenter_y = local_center_pix.y();

            int n_wide_before_mid_x = 0; //number of wide pixels in x/y before the cluster center, compensates the center shift due to expansion of wide rows/columns before the selected center
            int n_wide_before_mid_y = 0;
            for (int i = 0; i < n_double_x; ++i) {
                if (double_row[i] < mid_x) {
                    ++n_wide_before_mid_x;
                }
            }
            for (int i = 0; i < n_double_y; ++i) {
                if (double_col[i] < mid_y) {
                    ++n_wide_before_mid_y;
                }
            }
            int offset_x = TXSIZE/2 - mid_x - n_wide_before_mid_x; // compensate expansion shift from wide rows before the selected center, ensures that center pixel will end up at (TXSIZE/2, TYSIZE/2) in the input matrix after wide pixel expansion
            int offset_y = TYSIZE/2 - mid_y - n_wide_before_mid_y;

            if(Cluster_sizeX > TXSIZE or Cluster_sizeY > TYSIZE) continue;

            for (int i = 0; i < cluster.size(); ++i) {
                auto pix = cluster.pixel(i);
                int irow = int(pix.x) - row_offset + offset_x; // place the cluster center in the middle of the input matrix (TXSIZE/2, TYSIZE/2)
                int icol = int(pix.y) - col_offset + offset_y;

                if ((irow >= mrow+offset_x) || (icol >= mcol+offset_y)){
                    printf("irow or icol exceeded, SKIPPING. irow = %i, mrow = %i, offset_x = %i,icol = %i, mcol = %i, offset_y = %i\n",irow,mrow,offset_x,icol,mcol,offset_y);
                    continue;
                }
                clusbuf_temp[irow][icol] = float(pix.adc)/CHARGENORM; //pix.adc is actually in units of electrons
                Cluster_charge += float(pix.adc)/CHARGENORM;
            }

            int double_row_centered[double_pixel_buffer_size];
            int double_col_centered[double_pixel_buffer_size];
            for (int i = 0; i < double_pixel_buffer_size; ++i) {
                double_row_centered[i] = double_row[i] + offset_x;
                double_col_centered[i] = double_col[i] + offset_y;
            }

            //Expand double width rows
            int k=0,m=0;
            for(int i=0;i<TXSIZE;i++){
                if(m < n_double_x && i==double_row_centered[m]){
                    for(int j=0;j<TYSIZE;j++){
                        Cluster_raw[i][j]=clusbuf_temp[k][j]/2.;
                        Cluster_raw[i+1][j]=clusbuf_temp[k][j]/2.;
                    }
                    i++;
                    if(m==0 && n_double_x==2) {
                        double_row_centered[1]++; // If two rows are wide, they will be next to each other, so the second wide row will be right after the first one and needs to be shifted by 1 after the first one is expanded
                        m++;
                    } else if(m > 0) {
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

            // Set clusbuf to the original cluster with expanded rows
            for(int i=0;i<TXSIZE;i++){
                for(int j=0;j<TYSIZE;j++){
                    clusbuf_temp[i][j]=Cluster_raw[i][j];
                    Cluster_raw[i][j]=0.;
                }
            }

            // Expand double width columns
            for(int j=0;j<TYSIZE;j++){
                if(m < n_double_y && j==double_col_centered[m]){
                    for(int i=0;i<TXSIZE;i++){
                        Cluster_raw[i][j]=clusbuf_temp[i][k]/2.;
                        Cluster_raw[i][j+1]=clusbuf_temp[i][k]/2.;
                    }
                    j++;
                    if(m==0 && n_double_y==2) {
                        double_col_centered[1]++;
                        m++;
                    } else if(m > 0) {
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
            
            assert(cluster_max > 0);
            assert(cluster_max_2d > 0);
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

            if (genericCPE != nullptr) {
                auto const genericParams = genericCPE->getParameters(*clustp, *geomdetunit, ltp);
                Generic_x = std::get<0>(genericParams).x();
                Generic_y = std::get<0>(genericParams).y();
                Generic_xError = std::sqrt(std::max(0.f, std::get<1>(genericParams).xx()));
                Generic_yError = std::sqrt(std::max(0.f, std::get<1>(genericParams).yy()));
            }

            if (templateCPE != nullptr) {
                auto const templateParams = templateCPE->getParameters(*clustp, *geomdetunit, ltp);
                Template_x = std::get<0>(templateParams).x();
                Template_y = std::get<0>(templateParams).y();
                Template_xError = std::sqrt(std::max(0.f, std::get<1>(templateParams).xx()));
                Template_yError = std::sqrt(std::max(0.f, std::get<1>(templateParams).yy()));
            }

            //get sim hits

            std::vector<PSimHit> vec_simhits_assoc;
            std::unique_ptr<TrackerHitAssociator> associate(new TrackerHitAssociator(event,trackerHitAssociatorConfig_));
            vec_simhits_assoc.clear();
            vec_simhits_assoc = associate->associateHit(*pixhit);

            int iSimHit = 0;

            for (std::vector<PSimHit>::const_iterator sim_it = vec_simhits_assoc.begin();
                sim_it < vec_simhits_assoc.end() && iSimHit < SIMHITPERCLMAX; ++sim_it)
            {
                SimHit_x[iSimHit]    = ( sim_it->entryPoint().x() + sim_it->exitPoint().x() ) / 2.0;
                SimHit_y[iSimHit]    = ( sim_it->entryPoint().y() + sim_it->exitPoint().y() ) / 2.0;

                ++iSimHit;

            } // end sim hit loop
            nSimHit = iSimHit;
            for(int i = 0;i<nSimHit;i++){
                Generic_dx[i] = Generic_x - SimHit_x[i];
                Generic_dy[i] = Generic_y - SimHit_y[i];
                Generic_dxPull[i] = Generic_dx[i]/std::max(Generic_xError, 1.0e-12f);
                Generic_dyPull[i] = Generic_dy[i]/std::max(Generic_yError, 1.0e-12f);
                if (templateCPE != nullptr) {
                    Template_dx[i] = Template_x - SimHit_x[i];
                    Template_dy[i] = Template_y - SimHit_y[i];
                    Template_dxPull[i] = Template_dx[i]/std::max(Template_xError, 1.0e-12f);
                    Template_dyPull[i] = Template_dy[i]/std::max(Template_yError, 1.0e-12f);
                }
            }
            count++;
            out_Tree->Fill();
        }
    }

    printf("total cluster count = %i\n",count);

}
DEFINE_FWK_MODULE(ExtractCPEInfo);

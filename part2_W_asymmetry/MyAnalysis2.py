import ROOT 
from Samples2 import samp
import numpy as np

class MyAnalysis(object):
   
    def __init__(self, sample):

        """ The Init() function is called when an object MyAnalysis is initialised
        The tree corresponding to the specific sample is picked up 
        and histograms are booked.
        """
        self._tree = ROOT.TTree()        
        if(sample not in samp.keys() and sample != "data"):
            print("Error")
            exit
        self.histograms = {}
        self.sample = sample
        self._file = ROOT.TFile("files/"+sample+".root")
        self._file.cd()
        tree = self._file.Get("events")
        self._tree = tree
        self.nEvents = self._tree.GetEntries()
        print("Number of entries for " + self.sample + ": " + str(self.nEvents))
        
        # Book histograms
        self.bookHistos()

        # Parameters that might come in handy for some calculations
        self.nEventsPassW = 0  # Variable that counts the number of events we're left with (including weight)
        self.nEventsPass = 0 # Variable that counts the number of events we're left with (excluding weight)

    def getTree(self):
        return self._tree

    def getHistos(self):
        return self.histograms

    def bookHistos(self):
        h_nJet = ROOT.TH1F("NJet","#of jets", 6, -0.5, 6.5) # 0.5 because center must be at 0
        h_nJet.SetXTitle("%# of jets")
        self.histograms["NJet"] = h_nJet 

        h_nJetFinal = ROOT.TH1F("NJetFinal","#of jets", 6, -0.5, 6.5)
        h_nJetFinal.SetXTitle("%# of jets")
        self.histograms["NJetFinal"] = h_nJetFinal 

        h_MuonIso = ROOT.TH1F("Muon_Iso","Muon Isolation", 25, 0., 3.)
        h_MuonIso.SetXTitle("Muon Isolation")
        self.histograms["Muon_Iso"] = h_MuonIso 

        h_NIsoMu = ROOT.TH1F("NIsoMu","Number of isolated muons", 5, 0.5, 5.5)
        h_NIsoMu.SetXTitle("Number of isolated muons")
        self.histograms["NIsoMu"] = h_NIsoMu 

        h_MuonPt = ROOT.TH1F("Muon_Pt","Muon P_T", 50, 0., 200.)
        h_MuonPt.SetXTitle("Muon P_T")
        self.histograms["Muon_Pt"] = h_MuonPt 

        h_METpt = ROOT.TH1F("MET_Pt","MET P_T", 25, 0., 300.)
        h_METpt.SetXTitle("MET P_T")
        self.histograms["MET_Pt"] = h_METpt 

        h_JetPt = ROOT.TH1F("Jet_Pt","Jet P_T", 50, 0., 200.)
        h_JetPt.SetXTitle("Jet P_T")
        self.histograms["Jet_Pt"] = h_JetPt 

        h_JetBtag = ROOT.TH1F("Jet_Btag","Jet B tag", 10, 1., 6.)
        h_JetBtag.SetXTitle("Jet B tag")
        self.histograms["Jet_btag"] = h_JetBtag 

        h_NBtag = ROOT.TH1F("NBtag","Jet B tag", 4, 0.5, 4.5)
        h_NBtag.SetXTitle("Number of B tagged jets")
        self.histograms["NBtag"] = h_NBtag

        h_MuonpEta = ROOT.TH1F("Muon_eta_p","Muon+ eta", 500, 0., 0.4)
        h_MuonpEta.SetXTitle("Muon+ eta")
        self.histograms["Muon_eta_p"] = h_MuonpEta

        h_MuonmEta = ROOT.TH1F("Muon_eta_n","Muon- eta", 500, 0., 0.4)
        h_MuonmEta.SetXTitle("Muon- eta")
        self.histograms["Muon_eta_n"] = h_MuonmEta

        h_MET_p = ROOT.TH1F("MET_p","MET +", 50, -10., 170.)
        h_MET_p.SetXTitle("MET +")
        self.histograms["MET_p"] = h_MET_p

        h_MET_m = ROOT.TH1F("MET_n","MET -", 50, -10., 170.)
        h_MET_m.SetXTitle("MET -")
        self.histograms["MET_n"] = h_MET_m

        mWp = ROOT.TH1F("W_mass_p","Transverse W+ mass", 100, 0., 150.)
        mWp.SetXTitle("W mass (transverse)")
        self.histograms["W_mass_p"] = mWp

        mWm = ROOT.TH1F("W_mass_n","Transverse W- mass", 100, 0., 150.)
        mWm.SetXTitle("W mass (transverse)")
        self.histograms["W_mass_n"] = mWm

    def saveHistos(self):
        outfilename = self.sample + "_histos.root"
        outfile = ROOT.TFile(outfilename, "RECREATE")
        outfile.cd()
        for h in self.histograms.values():
            h.Write()
        outfile.Close()

    def processEvent(self, entry, cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high):
        tree = self.getTree()
        tree.GetEntry(entry)
        w = tree.EventWeight

        event_pass = True # Default: All events are passed

        # ---------------------- Cuts ----------------------

        # Cut 1: Number of isolated muons
        muonRelIsoCut = 0.1
        nIsoMu = 0
        for m in range(tree.NMuon):
            muon = ROOT.TLorentzVector(tree.Muon_Px[m],tree.Muon_Py[m],tree.Muon_Pz[m],tree.Muon_E[m])
            if (muon.Pt() > cut_muonPT and (tree.Muon_Iso[m]/muon.Pt()) < muonRelIsoCut):
                if cut_Eta_low < np.abs(muon.PseudoRapidity()) < cut_Eta_high:
                    nIsoMu += 1

        # Apply muon isolation cut:
        if not nIsoMu > 0:
            event_pass = False

        # Cut 2: MET
        MET = np.sqrt(tree.MET_px**2 + tree.MET_py**2)
        if MET < cut_MET:
            event_pass = False

        # Cut 3: Trigger
        if not tree.triggerIsoMu24 > 0:
            event_pass = False

        # ---------------------- Fill the Histograms ----------------------

        if not event_pass:
            return

        for m in range(tree.NMuon):
            muon = ROOT.TLorentzVector(tree.Muon_Px[m],tree.Muon_Py[m],tree.Muon_Pz[m],tree.Muon_E[m])
            muon_T = ROOT.TLorentzVector(tree.Muon_Px[m],tree.Muon_Py[m],0.,np.sqrt(tree.Muon_E[m]**2-tree.Muon_Pz[m]**2))
            MET_vec = ROOT.TLorentzVector(tree.MET_px, tree.MET_py,0.,MET)
            W_mass = (muon_T + MET_vec).M()
            if (muon.Pt() > cut_muonPT and (tree.Muon_Iso[m]/muon.Pt()) < muonRelIsoCut):
                if cut_Eta_low < np.abs(muon.PseudoRapidity()) < cut_Eta_high:
                    if tree.Muon_Charge[m] > 0:
                        self.histograms["Muon_eta_p"].Fill(np.abs(muon.PseudoRapidity()), w)
                        self.histograms["MET_p"].Fill(MET, w)
                        self.histograms["W_mass_p"].Fill(W_mass, w)
                    if tree.Muon_Charge[m] < 0:
                        self.histograms["Muon_eta_n"].Fill(np.abs(muon.PseudoRapidity()), w)
                        self.histograms["MET_n"].Fill(MET, w)
                        self.histograms["W_mass_n"].Fill(W_mass, w)

        self.nEventsPass += 1
        self.nEventsPassW += w

    # processEvents run the function processEvent on each event stored in the tree
    def processEvents(self, cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high):
        nevts = self.nEvents
        for i in range(nevts):
            self.processEvent(i, cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)
        self.saveHistos()


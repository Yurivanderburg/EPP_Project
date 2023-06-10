import ROOT 
from Samples1 import samp
import numpy as np

class MyAnalysis(object):
   
    def __init__(self, sample):

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
        self.nEventsPassW = 0   # Variable that counts the number of events we're left with (including weight)
        self.nEventsPass = 0  # Variable that counts the number of events we're left with (excluding weight)
        self.nTriggered = 0 # Variable that was used to estimate the trigger efficiency

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

    def saveHistos(self):
        outfilename = self.sample + "_histos.root"
        outfile = ROOT.TFile(outfilename, "RECREATE")
        outfile.cd()
        for h in self.histograms.values():
            h.Write()
        outfile.Close()

    def processEvent(self, entry, cut_nBTags, cut_NJets, cut_MET, cut_muonPT):
        tree = self.getTree()
        tree.GetEntry(entry)
        w = tree.EventWeight

        event_pass = True # Default state: All events are accepted

        # ---------------------- Cuts ----------------------

        # Cut 1: Number of isolated muons
        muonRelIsoCut = 0.05
        nIsoMu = 0
        for m in range(tree.NMuon):
            muon = ROOT.TLorentzVector(tree.Muon_Px[m],tree.Muon_Py[m],tree.Muon_Pz[m],tree.Muon_E[m])
            self.histograms["Muon_Iso"].Fill(tree.Muon_Iso[m], w)
            if (muon.Pt() > cut_muonPT and (tree.Muon_Iso[m]/muon.Pt()) < muonRelIsoCut):
                nIsoMu += 1
        # Apply muon isolation cut:
        if nIsoMu != 1:
            event_pass = False

        # Cut 2: Total number of Jets
        n_jets = tree.NJet
        if n_jets < cut_NJets:
            event_pass = False

        # Cut 3: Number of b-tagged jets
        nBTags = 0
        for k in range(tree.NJet):
            if tree.Jet_btag[k] > 2.0 and tree.Jet_ID[k]:
                nBTags += 1

        if nBTags < cut_nBTags:
            event_pass = False

        # Cut 4: Missing transverse energy
        MET = np.sqrt(tree.MET_px**2 + tree.MET_py**2)

        if MET < cut_MET:
            event_pass = False

        # Cut 5: Trigger
        if tree.triggerIsoMu24 != 1:
            event_pass = False
            self.nTriggered += 1

        # ---------------------- Fill the Histograms ----------------------

        if not event_pass:
            return

        for m in range(tree.NMuon):
            muon = ROOT.TLorentzVector(tree.Muon_Px[m],tree.Muon_Py[m],tree.Muon_Pz[m],tree.Muon_E[m])
            self.histograms["Muon_Iso"].Fill(tree.Muon_Iso[m], w)
            if (muon.Pt() > cut_muonPT and (tree.Muon_Iso[m]/muon.Pt()) < muonRelIsoCut):
                self.histograms["Muon_Pt"].Fill(muon.Pt(), w)
        self.histograms["NIsoMu"].Fill(nIsoMu, w)
        self.histograms["MET_Pt"].Fill(MET,w)
        self.histograms["NJet"].Fill(n_jets,w)
        self.histograms["NBtag"].Fill(nBTags, w)

        self.nEventsPass += 1
        self.nEventsPassW += w

    # processEvents run the function processEvent on each event stored in the tree
    def processEvents(self, cut_nBTags, cut_NJets, cut_MET, cut_muonPT):
        nevts = self.nEvents
        for i in range(nevts):
            self.processEvent(i, cut_nBTags, cut_NJets, cut_MET, cut_muonPT)
        self.saveHistos()
        print("Number of selected entries for " + self.sample + ": " + str(self.nEventsPass))

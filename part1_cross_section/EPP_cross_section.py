import ctypes
from MyAnalysis1 import MyAnalysis
from Plotter1 import plotVar, plotShapes, getSigHisto, getBkgHisto, getHisto
import numpy as np


# ---------------------- Parameters ----------------------

# Cuts
cut_nBTags = 1
cut_NJets = 4
cut_MET = 30
cut_muonPT = 20
cut_muonIso = 1

# Misc
cross_sec_th = [173.60, 11.7]
Luminosity = [50, 5]
trigger_eff = [0.87, 0.05]
MC_signal_nocut = [7928.61, np.sqrt(7928.61)]
run_process = True  # Whether the histograms should be created anew

samples = ["qcd", "zz", "wz", "ww", "single_top", "dy", "wjets", "ttbar"]
vars = ["Muon_Pt", "NBtag", "NJet", "NIsoMu", "MET_Pt"]


# ---------------------- Functions ----------------------

def calc_score(sig, bgr):
    score = sig / np.sqrt(sig + bgr)
    score_err = np.sqrt(
        ((np.sqrt(sig+bgr) - sig/(2*np.sqrt(sig+bgr)))/(sig+bgr))**2 * sig +
        sig/(2*(sig+bgr)**(3/2))**2 * bgr
    )
    return [score, score_err]


def calc_accuracy(MC_cut, MC_nocut):
    acc = MC_cut[0]/MC_nocut[0]
    acc_err = np.sqrt(
        (1/MC_nocut[0] * MC_cut[1])**2 + (MC_cut[0]/(MC_nocut[0]**2) * MC_nocut[1])**2
    )
    return [acc, acc_err]


def calc_cross_section(N, lumi, eff, acc):
    # N = L * sigma * a * e
    N_err = N[1]
    lumi_err = lumi[1]
    eff_err = eff[1]
    acc_err = acc[1]
    cross_sec = N[0]/(lumi[0]*eff[0]*acc[0])
    cross_sec_err = np.sqrt(
        (1/(lumi[0]*eff[0]*acc[0]) * N_err)**2 + (N[0]/((lumi[0]**2)*eff[0]*acc[0]) * lumi_err)**2
        + (N[0]/((eff[0]**2)*lumi[0]*acc[0]) * eff_err)**2 + (N[0]/((acc[0]**2)*eff[0]*lumi[0]) * acc_err)**2
    )
    return [cross_sec, cross_sec_err]


def calc_integral_and_error(histo):
    minBin = histo.GetXaxis().GetFirst()
    maxBin = histo.GetXaxis().GetLast()
    nDataErr = ctypes.c_double()
    nData = histo.IntegralAndError(minBin, maxBin, nDataErr)
    return nData, nDataErr.value


# ---------------------- Create Histograms and Plots ----------------------

if run_process:
    TT = MyAnalysis("ttbar")
    TT.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    DY = MyAnalysis("dy")
    DY.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    QCD = MyAnalysis("qcd")
    QCD.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    SingleTop = MyAnalysis("single_top")
    SingleTop.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    WJets = MyAnalysis("wjets")
    WJets.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    WW = MyAnalysis("ww")
    WW.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    ZZ = MyAnalysis("zz")
    ZZ.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    WZ = MyAnalysis("wz")
    WZ.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    Data = MyAnalysis("data")
    Data.processEvents(cut_nBTags, cut_NJets, cut_MET, cut_muonPT)

    for v in vars:
        print("Variable: ", v)
        ### plotShapes (variable, samples,logScale )
        plotShapes(v, samples,  True)
        ### plotVar(variable, samples,isData, logScale )
        plotVar(v, samples,  True, True)


# ---------------------- Cross-section calculation ----------------------

print("---------------- Integrate Signal, Background and Score -------------------")
for var in vars:
    (sig, sig_err) = calc_integral_and_error(getSigHisto(var))
    (bg, bg_err) = calc_integral_and_error(getBkgHisto(var, samples))
    print(f"Score ({var}): ", calc_score(sig, bg))


print("---------------- Subtract Background -------------------")
obs = np.array(calc_integral_and_error(getHisto('MET_Pt', 'data')))
print("Observations: ", obs)
bkg = np.array(calc_integral_and_error(getBkgHisto('MET_Pt', samples)))
print("Expected bkg: ", bkg)
N_signal = obs - bkg
print("Background subtracted number of events: ", N_signal)


print("---------------- Calculate Acceptance -------------------")
MC_signal_cut = np.array(calc_integral_and_error(getHisto('MET_Pt', 'ttbar')))
acceptance = calc_accuracy(MC_signal_cut, MC_signal_nocut)
print("Acceptance: ", acceptance)


print("---------------- Cross-section -------------------")
cross_sec_exp = calc_cross_section(N=N_signal, lumi=Luminosity, eff=trigger_eff, acc=acceptance)
print(cross_sec_exp)

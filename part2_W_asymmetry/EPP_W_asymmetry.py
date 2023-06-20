import ctypes

from MyAnalysis2 import MyAnalysis
from Plotter2 import plotVar, plotShapes, getSigHisto, getBkgHisto, getHisto
import numpy as np
import ROOT
import pandas as pd

# ---------------------- Parameters ----------------------

# Eta_cuts = [[0,0.4],[0.4,0.8],[0.8,1.5],[1.5,1.8],[1.8,2.1]] IF CHANGING THE ETA CUT -> ALSO CHANGE HISTO X-LIM
cut_MET = 20
cut_muonPT = 30
cut_Eta_low = 0
cut_Eta_high = 0.4
cross_sec_th = [173.60, 11.7]
W_mass_th = [80.4, 0.01]
run_process = True

samples = ["qcd", "zz", "wz", "ww", "single_top", "dy", "wjets", "ttbar"]
vars = ["Muon_eta_p", "MET_p", "W_mass_p", "Muon_eta_n", "MET_n", "W_mass_n"]


# ---------------------- Functions ----------------------

def calc_score(sig, bgr):
    score = sig / np.sqrt(sig + bgr)
    score_err = np.sqrt(
        ((np.sqrt(sig + bgr) - sig / (2 * np.sqrt(sig + bgr))) / (sig + bgr)) ** 2 * sig +
        sig / (2 * (sig + bgr) ** (3 / 2)) ** 2 * bgr
    )
    return [score, score_err]


def calc_integral_and_error(histo):
    minBin = histo.GetXaxis().GetFirst()
    maxBin = histo.GetXaxis().GetLast()
    nDataErr = ctypes.c_double()
    nData = histo.IntegralAndError(minBin, maxBin, nDataErr)
    return nData, nDataErr.value


def make_fits(charge=None, save_plots=False):
    # Check whether charge is positive or negative
    if charge == "positive":
        MC_background = getBkgHisto('W_mass_p', samples)
        MC_signal = getSigHisto('W_mass_p')
        W_mass_data = getHisto("W_mass_p", 'data')
    elif charge == "negative":
        MC_background = getBkgHisto('W_mass_n', samples)
        MC_signal = getSigHisto('W_mass_n')
        W_mass_data = getHisto("W_mass_n", 'data')
    else:
        raise SyntaxError("Variable charge must be either {negative} or {positive}")

    # Background fit (on MC simulation)
    c1 = ROOT.TCanvas("c1")
    fit_bkg = ROOT.TF1("fit_bkg", "[0]*(TMath::Erf((x-[1])/[2])+1.)", 0, 150)
    fit_bkg.SetParameters(10, 60, -10)
    fit_bkg.SetLineColor(12)
    MC_background.Fit(fit_bkg)
    MC_background.Draw("same")

    # Signal fit (on MC simulation)
    c2 = ROOT.TCanvas("c2")
    fit_sig = ROOT.TF1("fit_sig", "[0]*TMath::Gaus(x,[1],[2]) + [3]*TMath::Gaus(x,[4],[5])", 0, 150)
    fit_sig.SetParameters(500, 100, 10, 300, 50, 40)
    fit_sig.SetLineColor(12)
    MC_signal.Fit(fit_sig)
    MC_signal.Draw("same")

    if save_plots:
        c1.Update()
        c1.SaveAs(f"Background_fit{charge}_{cut_Eta_low}.pdf", "pdf")
        c2.Update()
        c2.SaveAs(f"Signal_fit{charge}_{cut_Eta_low}.pdf", "pdf")

    # Total fit (on CMS data)
    c3 = ROOT.TCanvas("c3")
    fit_bkg_sig = ROOT.TF1("fit_sig_bkg", "[0]*(TMath::Erf((x-[1])/[2]) +1.) + [3]*TMath::Gaus(x,[4],[5]) + "
                                          "[6]*TMath::Gaus(x,[7],[8])", 0, 150)
    for i in range(9):
        if i == 0:
            fit_bkg_sig.SetParameter(i, fit_bkg.GetParameter(i))
        if i <= 2:
            fit_bkg_sig.FixParameter(i, fit_bkg.GetParameter(i))
        else:
            fit_bkg_sig.SetParameter(i, fit_sig.GetParameter(i - 3))
    fit_bkg_sig.SetLineColor(6)
    fit_bkg_sig.SetNpx(1000)
    W_mass_data.Fit(fit_bkg_sig)

    # Total fit: Signal subplot -> abbreviated as fbss_sig
    fbss_sig = ROOT.TF1("fit_sig", "[0]*TMath::Gaus(x,[1],[2]) + [3]*TMath::Gaus(x,[4],[5])", 0, 150)
    for i in range(6):
        fbss_sig.SetParameter(i, fit_bkg_sig.GetParameter(i + 3))
    fbss_sig.SetLineColor(1)

    # Total fit: Background subplot -> abbreviated as fbss_bkg
    fbss_bkg = ROOT.TF1("fit_bkg", "[0]*(TMath::Erf((x-[1])/[2])+1.)", 0, 150)
    for i in range(3):
        fbss_bkg.SetParameter(i, fit_bkg_sig.GetParameter(i))
    fbss_bkg.SetLineColor(2)

    # Create legend
    leg = ROOT.TLegend(0.6, 0.8, 0.97, 0.97)
    leg.SetNColumns(2)
    leg.AddEntry(W_mass_data, 'data')
    leg.AddEntry(fit_bkg_sig, 'sig. + backgr.')
    leg.AddEntry(fbss_sig, 'signal')
    leg.AddEntry(fbss_bkg, 'background')

    # Draw everything
    W_mass_data.Draw("same")
    fit_bkg_sig.Draw("same")
    fbss_sig.Draw("same")
    fbss_bkg.Draw("same")
    leg.Draw()
    c3.Update()

    if save_plots:
        c3.SaveAs(f"sig_fit_bkg_{charge}_{cut_Eta_low}.pdf", "pdf")

    # Calculate N_mass
    # N_mass is determined from the signal subplot in the total fit
    # N_mass_err is determined from IntegralAndError from dataset (because TF1.IntegralError did not work...)
    N_mass_ = fbss_sig.Integral(fbss_sig.GetXmin(), fbss_sig.GetXmax())
    N_mass_err_ = calc_integral_and_error(W_mass_data)[1]
    N_mass = [N_mass_, N_mass_err_]
    print("N_mass: ", N_mass)

    return N_mass


def calc_asymmetry(N_pos, N_neg):
    N_p = N_pos[0]
    N_m = N_neg[0]
    sum = N_p + N_m

    A = (N_pos[0] - N_neg[0]) / sum
    A_err = np.sqrt(
        (((2 * N_neg[0]) / sum ** 2) * N_pos[1]) ** 2 + (((2 * N_pos[0]) / sum ** 2) * N_neg[1]) ** 2
    )
    return [A*100, A_err*100]


# ---------------------- Create Histograms and Plots ----------------------

if run_process:
    TT = MyAnalysis("ttbar")
    TT.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    DY = MyAnalysis("dy")
    DY.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    QCD = MyAnalysis("qcd")
    QCD.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    SingleTop = MyAnalysis("single_top")
    SingleTop.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    WJets = MyAnalysis("wjets")
    WJets.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    WW = MyAnalysis("ww")
    WW.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    ZZ = MyAnalysis("zz")
    ZZ.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    WZ = MyAnalysis("wz")
    WZ.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    Data = MyAnalysis("data")
    Data.processEvents(cut_MET, cut_muonPT, cut_Eta_low, cut_Eta_high)

    for v in vars:
        print("Variable: ", v)
        ### plotShapes (variable, samples,logScale )
        plotShapes(v, samples, False)
        ### plotVar(variable, samples,isData, logScale )
        plotVar(v, samples, True, False)


# ---------------------- Calculate the W-charge asymmetry ----------------------

N_p = make_fits("positive", save_plots=True)
N_m = make_fits("negative", save_plots=True)
asymmetry = calc_asymmetry(N_p, N_m)
print("W-charge asymmetry: ", asymmetry)


# ---------------------- Save the Eta-cuts and the asymmetry to json files ----------------------

df = pd.DataFrame({'Eta_cut_low': cut_Eta_low, 'Eta_cut_high': cut_Eta_high, 'N_p': [N_p], 'N_m': [N_m],
                   'asymmetry': [asymmetry]})
df.to_json(fr'Results_asymmetry_{cut_Eta_low}_{cut_Eta_high}.json')

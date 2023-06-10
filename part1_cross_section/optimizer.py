from MyAnalysis1 import MyAnalysis
import numpy as np
import pandas as pd


# ---------------------- Parameters ----------------------

cut_NJets = np.linspace(0,8,9)
cut_nBTags = np.linspace(0,3,4)
cut_MET = np.linspace(0,200,21)
cut_muonPT = np.linspace(0,200,21)
cut_muonIso = 1 # Fixed
df_results = pd.DataFrame({'cut_nBTags': 0, 'cut_NJets': 0, 'cut_MET': 0, 'cut_muonPT': 0, 'score': 0,
                           'score_err': 0, 'score_W': 0, 'score_W_err': 0},
                          index=[0])


# ---------------------- Functions ----------------------

def calc_score(sig, bgr):
    return sig / np.sqrt(sig + bgr)

def calc_score_err(sig, bgr):
    return np.sqrt(
        ((np.sqrt(sig+bgr) - sig/(2*np.sqrt(sig+bgr)))/(sig+bgr))**2 * sig +
        sig/(2*(sig+bgr)**(3/2))**2 * bgr
    )


# ---------------------- Run the thing (this takes quite some time... ----------------------

runner = 0
for cut1 in cut_nBTags:
    for cut2 in cut_NJets:
        for cut3 in cut_MET:
            for cut4 in cut_muonPT:
                print(f"Starting iteration {runner}")

                TT = MyAnalysis("ttbar")
                TT.processEvents(cut1, cut2, cut3, cut4)

                DY = MyAnalysis("dy")
                DY.processEvents(cut1, cut2, cut3, cut4)

                QCD = MyAnalysis("qcd")
                QCD.processEvents(cut1, cut2, cut3, cut4)

                SingleTop = MyAnalysis("single_top")
                SingleTop.processEvents(cut1, cut2, cut3, cut4)

                WJets = MyAnalysis("wjets")
                WJets.processEvents(cut1, cut2, cut3, cut4)

                WW = MyAnalysis("ww")
                WW.processEvents(cut1, cut2, cut3, cut4)

                ZZ = MyAnalysis("zz")
                ZZ.processEvents(cut1, cut2, cut3, cut4)

                WZ = MyAnalysis("wz")
                WZ.processEvents(cut1, cut2, cut3, cut4)

                Data = MyAnalysis("data")
                Data.processEvents(cut1, cut2, cut3, cut4)

                # Calculate unweighted and weighted scores
                signal = TT.nEventsPass
                background = (DY.nEventsPass + QCD.nEventsPass + SingleTop.nEventsPass + WJets.nEventsPass +
                              WW.nEventsPass + ZZ.nEventsPass + WZ.nEventsPass)
                score = calc_score(signal, background)

                signal_W = TT.nEventsPassW
                background_W = (DY.nEventsPassW + QCD.nEventsPassW + SingleTop.nEventsPassW + WJets.nEventsPassW +
                              WW.nEventsPassW + ZZ.nEventsPassW + WZ.nEventsPassW)
                score_W = calc_score(signal_W, background_W)

                # Calculate errors
                score_err = calc_score_err(signal, background)
                score_W_err = calc_score_err(signal_W, background_W)

                # Fill dataframe:
                df_temp = pd.DataFrame({'cut_nBTags': cut1, 'cut_NJets': cut2, 'cut_MET': cut3, 'cut_muonPT': cut4,
                                        'score': score, 'score_err': score_err, 'score_W': score_W,
                                        'score_W_err': score_W_err}, index=[0])
                df_results = pd.concat([df_results, df_temp], axis=0, ignore_index=True)
                df_results.to_json(r'scores/score.json')

                # Update counter
                runner += 1

    # Save a dataset for every MET/muonPT run as backup
        df_results.to_json(fr'scores/score_{runner}.json')

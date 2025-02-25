from cmath import log
import uproot as up
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema, TreeMakerSchema
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea import lookup_tools
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
ak.behavior.update(candidate.behavior)
import numpy as np
import argparse
import os
from sswwg_utils import common_helper as com
from sswwg_utils import analyze_helper as ana
#import matplotlib.pyplot as plt
#from coffea import hist as chist
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-y', '--year', help='which year to run: 2016, 2017, 2018', default='2018')
parser.add_argument('-t', '--type', help='which type: data, mc', choices=('data', 'mc'), default='data')
parser.add_argument('-n', '--number', help='id of the rootfile, used for the name of output file')
parser.add_argument('-f', '--file', help='dir of the file')
parser.add_argument('-o', '--output_dir', help='output abs directory')
args = parser.parse_args()

print(args.number)
print(args.year)
print(args.type)
print(args.file)

filename = args.file

if args.type=='data':
    isdata=True
elif args.type=='mc':
    isdata=False
else:
    pass

events = NanoEventsFactory.from_root(filename, schemaclass=NanoAODSchema).events()
if isdata==True:
    events['nevents'] = len(events)
else:
    events['nevents'] = np.sum(events.Generator.weight)
print('ntotal: %d'%len(events.run))

year = args.year
if isdata==True:
    lumi_mask = ana.get_lumi_mask(events, year)
    events = events[lumi_mask]

muons = events.Muon
electrons = events.Electron
leptons = ak.concatenate([electrons,muons],axis=1)
taus = events.Tau
jets = events.Jet
photons = events.Photon
fatjets = events.FatJet
MET = events.MET
puppimet = events.PuppiMET

events['nmuons'] = np.sum(ak.ones_like(muons.pt),axis=1)
events['nelectrons'] = np.sum(ak.ones_like(electrons.pt),axis=1)
events['njets'] = np.sum(ak.ones_like(jets.pt),axis=1)
events['nleps'] = events.nmuons+events.nelectrons
muons['pt_orig'] = muons.pt
muons['pt'],muons['pt_roccor_up'],muons['pt_roccor_down'] = ana.apply_rochester_correction(muons,data=isdata,year=year)
# electrons['corrected_pt'] = electrons.pt


################################################################################
# for muon ID, I'm not sure if the PF muon > WP-loose/ WP-medium how to define
################################################################################
loose_muon_ip = (muons.pt>5) & (abs(muons.eta)<2.4) & (abs(muons.dxy)<0.05) & (abs(muons.dz)<0.1)  & (muons.pfRelIso04_all<0.4) & (muons.sip3d < 8) & (muons.mediumId==1)
veto_muon_sel = (muons.pt>10) & (abs(muons.eta)<2.4)
loose_muon_sel = loose_muon_ip 
tight_muon_sel = (muons.pt>10) & (abs(muons.eta)<2.4) & (abs(muons.dxy)<0.05) & (abs(muons.dz)<0.1)  & (muons.pfRelIso04_all<0.4) & (muons.sip3d < 8) & (muons.mediumId==1) & (muons.mvaTTH > -0.6)

if isdata==True:
    pass
else:
    muons['is_real'] = (~np.isnan(ak.fill_none(muons.matched_gen.pt, np.nan)))*1

muons['isveto'] = veto_muon_sel
muons['isloose'] = loose_muon_sel
muons['istight'] = tight_muon_sel
muons['istightcharge'] = (muons.tightCharge>1)
muons['iselectron'] = ak.zeros_like(muons.pt)
muons['ismuon'] = ak.ones_like(muons.pt)
events['nveto_muons'] = np.sum(veto_muon_sel,axis=1)
events['nloose_muons'] = np.sum(loose_muon_sel,axis=1)
events['ntight_muons'] = np.sum(tight_muon_sel,axis=1)
veto_muons = muons[muons.isveto]
good_muons = muons[muons.istight]

##########################################################################
# for electron and muon ID, I Haven't added Deep Jet of nearby jet, 
#I Can't find this variable in nanoaod, I think it's about b-tagging??
##########################################################################
loose_elec_sel = (electrons.pt > 7) & (abs(electrons.eta) < 2.5) & (abs(electrons.dxy) < 0.05) &  (abs(electrons.dz) < 0.1) & (electrons.sip3d < 8) &  (electrons.pfRelIso03_all<0.4) & (electrons.lostHits <=1) & (electrons.mvaFall17V2noIso_WPL==1)
veto_elec_sel = loose_elec_sel 
tight_elec_barrel = ((electrons.pt > 10) & (abs(electrons.eta) < 2.5) & (abs(electrons.dxy) < 0.05) &  (abs(electrons.dz) < 0.1) & (electrons.sip3d < 8) &  (electrons.pfRelIso03_all<0.4) & (electrons.lostHits ==0) & (electrons.mvaFall17V2noIso_WPL==1) &  
(electrons.sieie < 0.011) & (electrons.hoe < 0.1) & (electrons.eInvMinusPInv > -0.04) & (electrons.convVeto ==1) & (electrons.mvaTTH > 0.25))
tight_elec_endcup =((electrons.pt > 10) & (abs(electrons.eta) < 2.5) & (abs(electrons.dxy) < 0.05) &  (abs(electrons.dz) < 0.1) & (electrons.sip3d < 8) &  (electrons.pfRelIso03_all<0.4) & (electrons.lostHits ==0) & (electrons.mvaFall17V2noIso_WPL==1) & 
(electrons.sieie < 0.030) & (electrons.hoe < 0.1) & (electrons.eInvMinusPInv > -0.04) & (electrons.convVeto ==1) & (electrons.mvaTTH > 0.25))
tight_elec_sel = tight_elec_barrel | tight_elec_endcup
electrons['isveto'] = veto_elec_sel
electrons['isloose'] = loose_elec_sel
electrons['istight'] = tight_elec_sel
electrons['istightcharge'] = (electrons.tightCharge>1)
electrons['iselectron'] = ak.ones_like(electrons.pt)
electrons['ismuon'] = ak.zeros_like(electrons.pt)
events['nveto_electrons'] = np.sum(veto_elec_sel,axis=1)
events['nloose_electrons'] = np.sum(loose_elec_sel,axis=1)
events['ntight_electrons'] = np.sum(tight_elec_sel,axis=1)
veto_electrons = electrons[electrons.isveto]
good_electrons = electrons[electrons.istight]

index = ak.argsort(leptons.pt, ascending=False)
leptons = leptons[index]




tight_tau_sel = ((taus.pt > 20) & (abs(taus.eta) < 2.3) & (abs(taus.dz) < 0.2) & (abs(taus.dxy) < 1000) & (taus.decayMode!=3) &
                 (taus.idDeepTau2017v2p1VSjet ==16) & (taus.idDeepTau2017v2p1VSmu >1) & (taus.idDeepTau2017v2p1VSe) >1)
tight_taus = taus[tight_tau_sel==1]
veto_tau_sel = (taus.pt > 20) & (abs(taus.eta) < 2.3) & (abs(taus.dz) < 0.2) & (taus.idDeepTau2017v2p1VSjet>>2 & 1) 
events['ntight_taus'] = np.sum(tight_tau_sel,axis=1)
events['nveto_taus'] = np.sum(veto_tau_sel,axis=1)

#########################################################
#for jets:  
#########################################################
if isdata==True:
    pass
else:
    jets['pt_orig'] = jets.pt
    jets['mass_orig'] = jets.mass
    jets['score'] = jets.btagDeepFlavB      ######  btag
    jets['is_real'] = (~np.isnan(ak.fill_none(jets.matched_gen.pt, np.nan)))*1
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    corrected_jets = ana.apply_jet_corrections('2018').build(jets, lazy_cache=events.caches[0])
    jesr_unc = [i for i in corrected_jets.fields if i.startswith("JES") or i.startswith("JER")]
    jets["pt"] = corrected_jets.pt
    jets["mass"] = corrected_jets.mass
    for ibr in jesr_unc:
        jets[f"pt_{ibr}_up"] = corrected_jets[ibr].up.pt
        jets[f"pt_{ibr}_down"] = corrected_jets[ibr].down.pt
        jets[f"mass_{ibr}_up"] = corrected_jets[ibr].up.mass
        jets[f"mass_{ibr}_down"] = corrected_jets[ibr].down.mass

    index = ak.argsort(jets.pt, ascending=False)
    jets = jets[index]

jets_drclean_elec = ana.is_clean(jets,electrons[electrons.istight],0.4)
jets_drclean_mu = ana.is_clean(jets,muons[muons.istight],0.4)
jets_drclean_leptons = jets_drclean_elec & jets_drclean_mu


#good_jets_sel = (jets_drclean_leptons) & (jets.pt > 30) & (abs(jets.eta)<4.7) & (jets.jetId>>1 & 1) & (jets.score<0.2783)
good_jets_sel = (jets_drclean_leptons) & (jets.pt > 20) & (abs(jets.eta)<2.4) & (jets.btagDeepFlavB<0.0490) # add btag (zero b-veto jets)
blike_jets_sel = (jets_drclean_leptons) & (jets.pt > 20) & (abs(jets.eta)<2.4) & (jets.btagDeepFlavB >= 0.0490)
#good_jets_sel = (jets_drclean_leptons) & (jets.pt > 30) & (abs(jets.eta)<4.7) & (jets.jetId>>1 & 1)  
good_jets_sel = good_jets_sel==1
jets['isgood'] = good_jets_sel
good_jets = jets[good_jets_sel]
blike_jets = jets[blike_jets_sel]

#####   btag   #####
if isdata==True:
    pass
else:
    flav = good_jets.hadronFlavour
    abseta = np.abs(good_jets.eta)
    pt = good_jets.pt
    good_jets['btagSF'], good_jets['btagSF_up'], good_jets['btagSF_down']  = ana.get_btagsf(flav, abseta, pt, year)
events['ngood_jets'] = np.sum(good_jets_sel,axis=1)

######MET definition
MET['pt_orig'] = MET.pt
MET['phi_orig'] = MET.phi
MET['pt_roccor'], MET['phi_roccor'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt,good_muons.phi,good_muons.pt_orig)
# consider the jer corr, please note: for jets, the pt_raw is the pt_orig, think about it
if isdata==True:
    pass
else:
    # the jer is applied after considering roccorr on Muon
    MET['pt'], MET['phi'] = ana.corrected_polar_met(MET['pt_roccor'],MET['phi_roccor'],good_jets["pt"],good_jets["phi"],good_jets["pt_orig"])
    # uncertainties
    MET['pt_roccor_up'], MET['phi_roccor_up'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt_roccor_up,good_muons.phi,good_muons.pt)
    MET['pt_roccor_down'], MET['phi_roccor_down'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt_roccor_down,good_muons.phi,good_muons.pt)
    MET['pt_UnclusteredEnergy_up'], MET['phi_UnclusteredEnergy_up'] = ana.corrected_polar_met(
        MET['pt'],
        MET['phi'],
        good_jets["pt"],
        good_jets["phi"],
        good_jets["pt"],
        (
            True,
            MET.MetUnclustEnUpDeltaX,
            MET.MetUnclustEnUpDeltaY,
        ),
    )


    MET['pt_UnclusteredEnergy_down'], MET['phi_UnclusteredEnergy_down'] = ana.corrected_polar_met(
        MET['pt'],
        MET['phi'],
        good_jets["pt"],
        good_jets["phi"],
        good_jets["pt"],
        (
            False,
            MET.MetUnclustEnUpDeltaX,
            MET.MetUnclustEnUpDeltaY,
        ),
        )
    for ibr in jesr_unc:
        MET[f"pt_{ibr}_up"], MET[f"phi_{ibr}_up"] = ana.corrected_polar_met(MET['pt'],MET['phi'],good_jets[f"pt_{ibr}_up"],good_jets["phi"],good_jets["pt"])
        MET[f"pt_{ibr}_down"], MET[f"phi_{ibr}_down"] = ana.corrected_polar_met(MET['pt'],MET['phi'],good_jets[f"pt_{ibr}_down"],good_jets["phi"],good_jets["pt"])
    
##### cuts
###########################################
###### HLT#################################
        


passSingleEle = events.HLT['Ele32_WPTight_Gsf_L1DoubleEG']
passSingleMu = events.HLT['IsoMu24']
passDiEle = (events.HLT['Ele23_Ele12_CaloIdL_TrackIdL_IsoVL'] | (events.HLT['DoubleEle25_CaloIdL_MW']))
passDiMu = events.HLT['Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8']
passMuEle = (events.HLT['Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL']) | (events.HLT['Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']) | (events.HLT['Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']) | (events.HLT['Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ']) | (events.HLT['DiMu9_Ele9_CaloIdL_TrackIdL_DZ']) | (events.HLT['Mu8_DiEle12_CaloIdL_TrackIdL_DZ'])
passTriEle = False
passTriMu = (events.HLT['TripleMu_10_5_5_DZ']) | (events.HLT['TripleMu_12_10_5'])
passTrigger = passDiEle | passDiMu | passMuEle | passTriEle | passTriMu | passSingleEle | passSingleMu

HLT_sel = passTrigger ==1
events = events[HLT_sel]
leptons = leptons[HLT_sel]
veto_muons = veto_muons[HLT_sel]
good_muons = good_muons[HLT_sel]
veto_electrons = veto_electrons[HLT_sel]
good_electrons = good_electrons[HLT_sel]
good_jets = good_jets[HLT_sel]
blike_jets = blike_jets[HLT_sel]
taus = taus[HLT_sel]
MET = MET[HLT_sel]
npassed = len(events.run)
print('npassed_HLT: %d'%npassed)




good_leptons = ak.concatenate([good_electrons, good_muons], axis=1)
veto_leptons = ak.concatenate([veto_muons,veto_electrons], axis = 1)
is_electron_pair = (ak.count(good_electrons.pt, axis=1) == 2) & (ak.count(good_muons.pt, axis=1) == 0 )
is_muon_pair = (ak.count(good_muons.pt, axis=1) == 2) & (ak.count(good_electrons.pt, axis=1) == 0)
Mz = 91.
MET_sel = MET.pt > 20

njet_sel = (ak.count(good_jets.pt,axis=1)>=3) & (ak.count(blike_jets.pt,axis=1)==0)

nlep_sel = (ak.count(good_leptons.pt,axis=1)==3) 

npassed = len(events.run)


total_sel_1 = nlep_sel
# total_sel = nlep_sel
events = events[total_sel_1]
leptons = leptons[total_sel_1]
veto_muons = veto_muons[total_sel_1]
good_muons = good_muons[total_sel_1]
veto_electrons = veto_electrons[total_sel_1]
good_electrons = good_electrons[total_sel_1]
good_jets = good_jets[total_sel_1]
taus = taus[total_sel_1]
MET = MET[total_sel_1]
npassed = len(events.run)
print('npassed_nlepsel: %d'%npassed)

leading_pt = leptons.pt[:, 0] > 25  # leading lepton pt > 25 GeV
# npassed = len(events.run)
# print('npassed_ptl1sel: %d'%npassed)
subleading_pt = leptons.pt[:, 1] > 15  # subleading lepton pt > 15 GeV
# npassed = len(events.run)
# print('npassed_ptl2sel: %d'%npassed)
####### for mll ##################
leptons = ak.zip({
            'pt':     leptons['pt'],
            'eta':    leptons['eta'],
            'phi':    leptons['phi'],
            'mass':   leptons['mass'], 
        }, with_name="PtEtaPhiMLorentzVector")

nlep_sel_2 = leading_pt & subleading_pt 
nlep_sel_2 = nlep_sel_2[:len(events)]  
events = events[nlep_sel_2]
leptons = leptons[nlep_sel_2]
veto_muons = veto_muons[nlep_sel_2]
good_muons = good_muons[nlep_sel_2]
veto_electrons = veto_electrons[nlep_sel_2]
good_electrons = good_electrons[nlep_sel_2]
good_jets = good_jets[nlep_sel_2]
taus = taus[nlep_sel_2]
MET = MET[nlep_sel_2]
npassed = len(events.run)
print('npassed_leptonptsel: %d'%npassed)

###################################
#choose OSSF lepton pairs
###################################
mumu_sel = (
            (events.ntight_muons==2) & (np.sum(good_muons.charge,axis=1)==0)) #nleptons requirement
        
ee_sel = (
            (events.ntight_electrons==2) & (np.sum(good_electrons.charge,axis=1)==0)) #nleptons requirement
lepton_pair_sel = (mumu_sel | ee_sel)
lepton_pair_sel = lepton_pair_sel[:len(events)]  # 截取到与events长度一致

events = events[lepton_pair_sel]
leptons = leptons[lepton_pair_sel]
veto_muons = veto_muons[lepton_pair_sel]
good_muons = good_muons[lepton_pair_sel]
veto_electrons = veto_electrons[lepton_pair_sel]
good_electrons = good_electrons[lepton_pair_sel]
good_jets = good_jets[lepton_pair_sel]
taus = taus[lepton_pair_sel]
MET = MET[lepton_pair_sel]

npassed = len(events.run)

print('npassed_leptonpair_OSSF: %d'%npassed)

################ mll > 12GeV

lepton1 = leptons[:,0]
lepton2 = leptons[:,1]
lepton3 = leptons[:,2]
mll_1 = (lepton1 + lepton2).mass 
mll_2 = (lepton2 + lepton3).mass 
mll_3 = (lepton1 + lepton3).mass 
mll_cut_1 = (mll_1 > 12) & (mll_2 > 12) & (mll_3 > 12)

nlep_sel_3 =  mll_cut_1 
nlep_sel_3 = nlep_sel_3[:len(events)]  

events = events[nlep_sel_3]
leptons = leptons[nlep_sel_3]
veto_muons = veto_muons[nlep_sel_3]
good_muons = good_muons[nlep_sel_3]
veto_electrons = veto_electrons[nlep_sel_3]
good_electrons = good_electrons[nlep_sel_3]
good_jets = good_jets[nlep_sel_3]
taus = taus[nlep_sel_3]
MET = MET[nlep_sel_3]
npassed = len(events.run)
print('npassed_mllsel_12: %d'%npassed)


########### 25< mll < 100GeV
mll_cut_2_1 = (mll_1 > 25) & (mll_1 < 100)
mll_cut_2_2 = (mll_2 > 25) & (mll_2 < 100)
mll_cut_2_3 = (mll_3 > 25) & (mll_3 < 100)


nlep_sel_4 =  (mll_cut_2_1 |mll_cut_2_2|mll_cut_2_3) 
nlep_sel_4 = nlep_sel_4[:len(events)]  

events = events[nlep_sel_4]
leptons = leptons[nlep_sel_4]
veto_muons = veto_muons[nlep_sel_4]
good_muons = good_muons[nlep_sel_4]
veto_electrons = veto_electrons[nlep_sel_4]
good_electrons = good_electrons[nlep_sel_4]
good_jets = good_jets[nlep_sel_4]
taus = taus[nlep_sel_4]
MET = MET[nlep_sel_4]
npassed = len(events.run)
print('npassed_mllsel_25-100: %d'%npassed)


total_sel_3 =  MET_sel
total_sel_3 = total_sel_3[:len(events)] 

events = events[total_sel_3]
leptons = leptons[total_sel_3]
veto_muons = veto_muons[total_sel_3]
good_muons = good_muons[total_sel_3]
veto_electrons = veto_electrons[total_sel_3]
good_electrons = good_electrons[total_sel_3]
good_jets = good_jets[total_sel_3]
taus = taus[total_sel_3]
MET = MET[total_sel_3]
npassed = len(events.run)
print('npassed_METsel: %d'%npassed)

total_sel_2 =  njet_sel 
total_sel_2 = total_sel_2[:len(events)]  # 确保长度匹配

events = events[total_sel_2]
leptons = leptons[total_sel_2]
veto_muons = veto_muons[total_sel_2]
good_muons = good_muons[total_sel_2]
veto_electrons = veto_electrons[total_sel_2]
good_electrons = good_electrons[total_sel_2]
good_jets = good_jets[total_sel_2]
taus = taus[total_sel_2]
MET = MET[total_sel_2]
npassed = len(events.run)
print('npassed_njetsel: %d'%npassed)

lepton_bool_list = ['is_real','isloose','istight','istightcharge','iselectron','ismuon']

if npassed > 0:
    eve_dict = {}
    # basic info
    eve_dict['run'] = events.run
    eve_dict['luminosityBlock'] = events.luminosityBlock
    eve_dict['event'] = events.event
   
      
    eve_dict['nveto_muons'] = events.nveto_muons
    eve_dict['nveto_electrons'] = events.nveto_electrons
    eve_dict['nevents'] = events.nevents
    eve_dict['ntight_taus'] = events.ntight_taus
    eve_dict['nveto_taus'] = events.nveto_taus
    # other info for MC
    if isdata==True:
        pass
    else:
        eve_dict['Generator_weight'] = events.Generator.weight
        try:
            eve_dict['nLHEPdfWeight'] = ak.count(events.LHEPdfWeight,axis=-1)
            eve_dict['LHEPdfWeight'] = events.LHEPdfWeight
        except:
            pass
        try:
            eve_dict['nLHEScaleWeight'] = ak.count(events.LHEScaleWeight,axis=-1)
            eve_dict['LHEScaleWeight'] = events.LHEScaleWeight
        except:
            pass
        try:
            eve_dict['nLHEReweightingWeight'] = ak.count(events.LHEReweightingWeight,axis=-1)
            eve_dict['LHEReweightingWeight'] = events.LHEReweightingWeight
        except:
            pass
        try:
            eve_dict['nPSWeight'] = ak.count(events.PSWeight,axis=-1)
            eve_dict['PSWeight'] = events.PSWeight
        except:
            pass

        # pu weights
        eve_dict['PUWeight_nominal'], eve_dict['PUWeight_up'], eve_dict['PUWeight_down'] = ana.get_pusf(events.Pileup.nTrueInt, year)
        # genpart
        try:
            eve_dict['nGenPart'] = ak.count(events.GenPart.pt,axis=-1)
        except:
            pass
        for ibr in events.GenPart.fields:
            if not ibr.endswith('IdxMotherG') and not ibr.endswith('IdxG'):
                eve_dict[f"GenPart_{ibr}"] = events.GenPart[ibr]
        # genjet
        eve_dict['nGenJet'] = ak.count(events.GenJet.pt,axis=-1)
        for ibr in events.GenJet.fields:
            if ibr=='hadronFlavour':
                eve_dict[f"GenJet_{ibr}"] = ak.values_astype(events.GenJet[ibr], np.int32)
            else:
                eve_dict[f"GenJet_{ibr}"] = events.GenJet[ibr]
        # GenDressedLepton
        eve_dict['nGenDressedLepton'] = ak.count(events.GenDressedLepton.pt,axis=-1)
        for ibr in events.GenDressedLepton.fields:
            if ibr == 'hasTauAnc':
                eve_dict[f"GenDressedLepton_{ibr}"] = events.GenDressedLepton[ibr]*1
            else:               
                eve_dict[f"GenDressedLepton_{ibr}"] = events.GenDressedLepton[ibr]
        
    # muon info
#                     eve_dict['nMuon'] = ak.count(good_muons.pt,axis=-1)
    for ibr in veto_muons.fields:
        if ibr in lepton_bool_list:
            eve_dict[f'Muon_{ibr}'] = veto_muons[ibr]*1
        else:
            eve_dict[f'Muon_{ibr}'] = veto_muons[ibr]
    # electron info
#                     eve_dict['nElectron'] = ak.count(good_electrons.pt,axis=-1)
    for ibr in veto_electrons.fields:
        if ibr in lepton_bool_list:
            eve_dict[f'Electron_{ibr}'] = veto_electrons[ibr]*1
        else:
            eve_dict[f'Electron_{ibr}'] = veto_electrons[ibr]
    
    # jet info
    eve_dict['nJet'] = ak.count(good_jets.pt,axis=-1)
    for ibr in good_jets.fields:
        if not ibr in ['muonIdxG','electronIdxG']:
            if 'isgood' in ibr:
                eve_dict[f'Jet_{ibr}'] = good_jets[ibr]*1
            else:
                eve_dict[f'Jet_{ibr}'] = good_jets[ibr]
            
    # tau info
    for ibr in taus.fields:
        eve_dict[f'Tau_{ibr}'] = taus[ibr]
    # met info
    for ibr in MET.fields:
        eve_dict[f'MET_{ibr}'] = MET[ibr]
    # puppimet info
    for ibr in events.PuppiMET.fields:
        eve_dict[f'PuppiMET_{ibr}'] = events.PuppiMET[ibr]
    for ibr in events.fields:
        if ibr.startswith('fixedGridRho'):
            eve_dict[f'{ibr}'] = events[ibr]
    eve_ak = ak.Array(eve_dict) # this will make the store step much faster
    ak.to_parquet(eve_ak,f"{args.output_dir}/{args.number}.parquet")
#                 ak.to_parquet(eve_ak,'/data/pubfs/tyyang99/jupyter_files/pkutree/ssww_events/'+'test'+f'_{com.get_randomstr()}'+".parquet")
else:
    reminder = open(f'{args.output_dir}/No_event_reminder!','w')
    reminder.close()








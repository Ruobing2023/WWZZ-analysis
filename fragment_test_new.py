import ROOT
import array
import sys

ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")

class GenPart:
    def __init__(self, pt, eta, phi, mass, status, statusFlags, genPartIdxMother, pdgId, idx):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.status = status
        self.statusFlags = statusFlags
        self.genPartIdxMother = genPartIdxMother
        self.pdgId = pdgId
        self.idx = idx

def find_daughters(parent, collection):
    daughters = []
    for particle in collection:
        if particle.genPartIdxMother == parent.idx:
            daughters.append(particle)
    return daughters


MAX_INSTANCES = 4000
IS_LAST_COPY = (1 << 13)

# Define arrays to store the data
nGenPart = array.array('I', [0])
GenPart_pt = array.array('f', [0.0] * MAX_INSTANCES)
GenPart_eta = array.array('f', [0.0] * MAX_INSTANCES)
GenPart_phi = array.array('f', [0.0] * MAX_INSTANCES)
GenPart_mass = array.array('f', [0.0] * MAX_INSTANCES)
GenPart_status = array.array('i', [0] * MAX_INSTANCES)
GenPart_statusFlags = array.array('i', [0] * MAX_INSTANCES)
GenPart_genPartIdxMother = array.array('i', [0] * MAX_INSTANCES)
GenPart_pdgId = array.array('i', [0] * MAX_INSTANCES)


fn = sys.argv[1]
fp = ROOT.TFile.Open(fn, 'read')


tree = fp.Get("Events")


tree.SetBranchAddress("nGenPart", nGenPart)
tree.SetBranchAddress("GenPart_pt", GenPart_pt)
tree.SetBranchAddress("GenPart_eta", GenPart_eta)
tree.SetBranchAddress("GenPart_phi", GenPart_phi)
tree.SetBranchAddress("GenPart_mass", GenPart_mass)
tree.SetBranchAddress("GenPart_status", GenPart_status)
tree.SetBranchAddress("GenPart_statusFlags", GenPart_statusFlags)
tree.SetBranchAddress("GenPart_genPartIdxMother", GenPart_genPartIdxMother)
tree.SetBranchAddress("GenPart_pdgId", GenPart_pdgId)

tree.SetBranchStatus("*", 0)
tree.SetBranchStatus("*GenPart*", 1)

nof_entries = tree.GetEntries()
counter = {}

nof_Higgs = 0 
nof_qq_events = 0  
nof_HH_events = 0  

for idx in range(nof_entries):
    tree.GetEntry(idx)

  
    genParts = []
    for i in range(nGenPart[0]):
        genParts.append(GenPart(
            GenPart_pt[i], GenPart_eta[i], GenPart_phi[i], GenPart_mass[i],
            GenPart_status[i], GenPart_statusFlags[i], GenPart_genPartIdxMother[i],
            GenPart_pdgId[i], i
        ))

    # Check for Higgs boson
    has_Higgs = any(p.pdgId == 25 for p in genParts)
    if has_Higgs:
        nof_Higgs += 1

    genW = []
    genZ = []
    for genPart in genParts:
        if abs(genPart.pdgId) == 24 and genPart.statusFlags & IS_LAST_COPY:  # W+ or W-
            genW.append(genPart)
        elif abs(genPart.pdgId) == 23 and genPart.statusFlags & IS_LAST_COPY:  # Z
            genZ.append(genPart)

    # Check we found exactly 2 W's and 2 Z's
    assert(len(genW) == 2), f"Expected 2 W's, found {len(genW)}"
    assert(len(genZ) == 2), f"Expected 2 Z's, found {len(genZ)}"

    # Determine parent of WWZZ
    WWZZ_parents = set(p.genPartIdxMother for p in (genW + genZ) if p.genPartIdxMother >= 0)
    WWZZ_from_Higgs = any(genParts[parent].pdgId == 25 for parent in WWZZ_parents)

    if WWZZ_from_Higgs:
        nof_HH_events += 1
    else:
        nof_qq_events += 1

    # Find the daughters of WWZZ
    label = ''
    nof_ele, nof_mu, nof_quarks, nof_neutrinos, nof_tau = 0, 0, 0, 0, 0


    wDaughters = []
    for genWParticle in genW:
        wDaughters.extend(find_daughters(genWParticle, genParts))
    
    # Analyze the daughters of Z's
    zDaughters = []
    for genZParticle in genZ:
        zDaughters.extend(find_daughters(genZParticle, genParts))

    # Count the numbers
    all_daughters = wDaughters + zDaughters
    for daughter in all_daughters:
        if abs(daughter.pdgId) == 11:  # Electron
            nof_ele += 1
        elif abs(daughter.pdgId) == 13:  # Muon
            nof_mu += 1
        elif abs(daughter.pdgId) == 15:  # Tau
            nof_tau += 1
        elif abs(daughter.pdgId) in [1, 2, 3, 4, 5]:  # Quarks
            nof_quarks += 1
        elif abs(daughter.pdgId) in [12, 14, 16]:  # Neutrinos
            nof_neutrinos += 1

    label = ":{} ele {} mu {} Q {} Nu {} Tau".format(nof_ele, nof_mu, nof_quarks, nof_neutrinos, nof_tau)

    if label not in counter:
        counter[label] = 0
    counter[label] += 1
fp.Close()


print("Processed {} event(s) from file {}".format(nof_entries, fn))
print(f"Total number of events in file: {nof_entries}")
print(f"Number of Higgs bosons: {nof_Higgs}")
print(f"Number of WWZZ from qq: {nof_qq_events}")
print(f"Number of WWZZ from HH: {nof_HH_events}")

for entry in sorted(counter.items(), key=lambda x: x[1], reverse=True):
    print("{} -> {} ({:.2f}%)".format(entry[0], entry[1], entry[1] / nof_entries * 100))








import mdtraj as md
from MI_fluc import *
traj = md.load('EAG_withnoCAM_pro.dcd', top='EAGwithnoCaM_pro.psf')
flucuationMI(traj,"fluctMI")

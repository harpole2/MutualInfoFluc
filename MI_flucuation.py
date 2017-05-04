import mdtraj as md
from MI_fluc import *
traj = md.load('TRAJ', top='TOPOLOGY')
flucuationMI(traj,"fluctMI")

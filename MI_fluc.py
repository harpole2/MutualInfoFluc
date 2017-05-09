import mdtraj as md
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import skew

#traj = md.load('EAG_withnoCAM_pro.dcd', top='EAGwithnoCaM_pro.psf')

def flucuationMI(traj,filename):
	nFrames = int(traj.n_frames)
	sel=traj.topology.select('protein and backbone')
	traj.superpose(traj,frame=0,atom_indices=sel)
	nResidues = int(traj.n_residues)
		
	points = np.zeros((nFrames,nResidues,3))
	# Store residue CApoints points
	for i in range(0,nResidues):
		tmpTraj_ind = traj.topology.select('protein and resid ' + str(i) + ' and name CA')
		tmpTraj = traj.atom_slice(tmpTraj_ind)
		tmpPoints = tmpTraj.xyz
		points[::,i,::] = np.mean(tmpPoints,axis=1)

	#get the mean position of each point
	mean_points = np.mean(points,axis=0)

	fluc_points = np.zeros((nFrames,nResidues))
	for i in range(0, fluc_points.shape[0]):
		for j in range(0,fluc_points.shape[1]):
			fluc_points[i,j]=np.linalg.norm(points[i,j]-mean_points[j])

	#calculate number of bins using doannes rule
	sha = fluc_points.shape[0]
	g1=np.atleast_1d(skew(fluc_points))
	sg1 = np.sqrt(6.0 * (sha - 2) / ((sha + 1) * (sha + 3)))
	bins= np.floor(1 + np.log2(sha) + np.log2(1 + abs(g1)/sg1))

	MI=np.zeros((nResidues,nResidues))
	for i in range(0,nResidues):
		for j in range(i,nResidues):
			minbin=min(bins[i],bins[j])
			hist2d=np.histogram2d(fluc_points[:,i],fluc_points[:,j],bins=minbin,normed=True)[0]
			MI[i,j]=mutual_info_score(None, None, contingency=hist2d)
			MI[j,i]=MI[i,j]

	np.savetxt(filename+'.txt',MI)

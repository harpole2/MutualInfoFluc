import mdtraj as md
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import skew

def MIchi1(traj,filename):
	chi1=md.compute_chi1(traj,periodic=False)[1]
	nFrames = int(traj.n_frames)
	nResidues = int(traj.n_residues)

	chitot = np.zeros((nFrames,nResidues))
	#find all ALA and GLY to make dimensions of zeroes
	table, bonds = traj.topology.to_dataframe()
	test1=table[table['resName'] == 'ALA']
	res1=test1.resSeq.unique()-1
	test2=table[table['resName'] == 'GLY']
	res2=test2.resSeq.unique()-1
	mergres=np.concatenate((res1,res2),axis=0)

	count=0
	for i in range(0,nResidues):
		if i in mergres:
			continue
		else:
			chitot[:,i]=chi1[:,count]
			count = count +1
		



	#calculate number of bins using doannes rule
	sha = chitot.shape[0]
	g1=np.atleast_1d(skew(chitot))
	sg1 = np.sqrt(6.0 * (sha - 2) / ((sha + 1) * (sha + 3)))
	bins= np.floor(1 + np.log2(sha) + np.log2(1 + abs(g1)/sg1))

	MI=np.zeros((nResidues,nResidues))
	for i in range(0,chitot.shape[1]):
		for j in range(i,chitot.shape[1]):
			if sum(chitot[:,i]) == 0:
				MI[i,j]=0
				MI[j,i]=0 
			elif sum(chitot[:,j]) == 0:
				MI[i,j]=0
				MI[j,i]=0 
			else:
				minbin=min(bins[i],bins[j])
				hist2d=np.histogram2d(chitot[:,i],chitot[:,j],bins=minbin,normed=True)[0]
				MI[i,j]=mutual_info_score(None, None, contingency=hist2d)
				MI[j,i]=MI[i,j]

	np.savetxt(filename+'chi1_ALA_GLY_zero.txt',MI)

#%%
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import pylab as plt
import os
import numpy as np

work_dir = ".\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Desktop\\Demo\\"
inpfile = 'Gabbs4_Homog_input.inp'
out_dir = "SimPEG_GRAV_Homo_Inv\\"
dsep = '\\'
dsep = os.path.sep

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)


# Read input file
driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

m0 = driver.m0
mgeo = driver.mref

# Get unique geo units
geoUnits = np.unique(mgeo).tolist()


# Build a dictionary for the units
mstart = np.asarray([np.median(m0[mgeo==unit]) for unit in geoUnits])

#actv = mrho!=-100

# Build list of indecies for the geounits
index = []
for unit in geoUnits:
#    if unit!=0:
    index += [mgeo==unit]
nC = len(index)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
homogMap = Maps.HomogeneousMap(index)



#ndata = survey.srcField.rxList[0].locs.shape[0]
#
#actv = driver.activeCells
#nC = len(actv)

# Create static map
#static = driver.staticCells
#dynamic = driver.dynamicCells
#
#staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)


#%% Plot obs data
# PF.Gravity.plot_obs_2D(survey.srcField.rxList[0].locs, survey.dobs,'Observed Data')

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=homogMap, actInd=actv)

survey.pair(prob)

# Load weighting  file
if driver.wgtfile is None:
    # wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3., np.min(mesh.hx)/4.)
    # wr = wr**2.

    # Make depth weighting
    wr = np.zeros(nC)
    for ii in range(survey.nD):
        wr += ((prob.F[ii, :]*prob.rhoMap.deriv(m0))/survey.std[ii])**2.
#    
#    scale = Utils.mkvc(homogMap.P.sum(axis=0))
#    for ii in range(nC):
#        wr[ii] *= scale[ii]
    wr = (wr/np.max(wr))
    wr = wr

else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, work_dir + dsep + wgtfile)
    wr = wr[actv]
    wr = wr**2.



Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.den',
                              actvMap*(homogMap.P*wr))

idenMap = Maps.IdentityMap(nP=nC)
regMesh = Mesh.TensorMesh([nC])

# Create a regularization
reg = Regularization.Sparse(regMesh, mapping=idenMap)
reg.norms = driver.lpnorms

if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

reg.cell_weights = wr#driver.cell_weights*mesh.vol**0.5
reg.mref = mstart

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Write out the predicted file and generate the forward operator
#pred = prob.fields(m0)
#
#PF.Gravity.writeUBCobs(work_dir + dsep + out_dir + '\\Pred0.dat',survey,pred)

opt = Optimization.ProjectedGNCG(maxIter=30, lower=driver.bounds[0],
                                 upper=driver.bounds[1], 
                                 maxIterLS = 50, maxIterCG= 30, 
                                 tolCG = 1e-4)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1.)
IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=2)
update_Jacobi = Directives.UpdateJacobiPrecond()
#saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
#saveModel.fileName = work_dir + dsep + out_dir + 'GRAV'

saveDict = Directives.SaveOutputDictEveryIteration()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS, saveDict,
                                                      update_Jacobi])
# Run inversion
mrec = inv.run(mstart)

# Plot predicted
pred = prob.fields(mrec)

# PF.Gravity.plot_obs_2D(survey, 'Observed Data')
print("Final misfit:" + str(np.sum(((survey.dobs-pred)/survey.std)**2.)))

# Write result
if getattr(invProb, 'l2model', None) is not None:

    m_l2 = actvMap*homogMap*invProb.l2model
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_l2l2.den', m_l2)

m_out = actvMap*homogMap*mrec
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_lplq.den', m_out)

#PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_l2.pre',
#                         survey, d=survey.dpred(invProb.l2model))

PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_lp.pre',
                         survey, d=invProb.dpred)
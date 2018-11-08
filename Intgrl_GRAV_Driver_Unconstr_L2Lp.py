#%%
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import pylab as plt
import os
import numpy as np

work_dir = '.\\'
inpfile = 'Gabbs5_FWD_input.inp'
out_dir = "SimPEG_GRAV_Inv\\"
dsep = '\\'
dsep = os.path.sep

os.system('mkdir ' + work_dir + dsep + out_dir)

# Read input file
driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
mesh = driver.mesh
survey = driver.survey
wd = survey.std

# Create active map to go from reduce set to full
actv = driver.activeCells
nC = len(actv)
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create static map
static = driver.staticCells
dynamic = driver.dynamicCells

staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)
mstart = driver.m0[dynamic]
mref = driver.mref[dynamic]

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=staticCells, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Create misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./wd

# Write out the predicted file and generate the forward operator
pred = prob.fields(mstart)
PF.Gravity.writeUBCobs(work_dir + dsep + out_dir + '\\Pred0.dat', survey, pred)

# Load weighting  file
if driver.wgtfile is None:
    # wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3., np.min(mesh.hx)/4.)
    # wr = wr**2.

    # Make depth weighting
#    wr = np.sum((dmis.W*prob.F)**2., axis=0)**0.5
    wr = np.zeros(nC)
    for ii in range(survey.nD):
        wr += (prob.F[ii, :]/survey.std[ii])**2.

    wr = (wr/np.max(wr))

else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, work_dir + dsep + wgtfile)
    wr = wr[actv]
    wr = wr**2.

# % Create inversion objects
# Starting with regularization
reg = Regularization.Sparse(mesh, indActive=actv,
                            mapping=staticCells, gradientType='total')
reg.mref = driver.mref[dynamic]
reg.cell_weights = wr
reg.norms = driver.lpnorms
if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

# Optimization function
opt = Optimization.ProjectedGNCG(maxIter=100, lower=driver.bounds[0],
                                 upper=driver.bounds[1],
                                 maxIterLS=50, maxIterCG=10, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=2)
update_Jacobi = Directives.UpdateJacobiPrecond()
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + dsep + out_dir + 'GRAV'

saveDict = Directives.SaveOutputDictEveryIteration()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS, saveDict,
                                                      update_Jacobi,
                                                      saveModel])
# Run inversion
mrec = inv.run(mstart)

# Plot predicted
pred = prob.fields(mrec)

# Save last models and predicted data
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_l2l2.den', actvMap*staticCells*invProb.l2model)
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_lplq.den', actvMap*staticCells*mrec)

PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_l2.pre',
                       survey, d=survey.dpred(invProb.l2model))

PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_lp.pre',
                       survey, d=survey.dpred(invProb.model))

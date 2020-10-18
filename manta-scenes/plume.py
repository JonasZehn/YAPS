### Run using Mantaflow (http://mantaflow.com/)
### Tested with Mantaflow commit of 18th of May, 2018
### Usage: ./build/manta scenes/plume.py (sf|mc) [r]
### Advector name:
###     sf = Stable Fluids/First order advector
###     mc = MacCormack/Second order advector
### Method name:
###     r = use Advection-Reflection method
### build is the build directory, scenes is the directory where this file is stored

from manta import *
import os
import sys

#command line argument treatment:
try:
    if len(sys.argv) < 2:
        raise ValueError('Too few arguments supplied, supply an advector name: either sf or mc')
    
    advectorName = sys.argv[1]
    if advectorName== "sf":
        advectionOrder = 1
    elif advectorName=="mc":
        advectionOrder = 2
    else: raise ValueError('Unrecognized advector name "' + advectorName + '"')

    if len(sys.argv) < 3:
        methodName = ""
        newMethod = false
    else:
        methodName = sys.argv[2]
        if methodName == "r":
            newMethod = True
        else: raise ValueError('Method name "' + methodName + '" not recognized, only option is none or r')
        
except Exception as e:
    print(str(e))
    print("Usage: manta scene/plume.py (sf|mc) [r]")
    sys.exit(1)

# solver params
res = 128
gs  = vec3(res, 2*res, res)
s   = FluidSolver(name='main', gridSize = gs)

# prepare grids
flags    = s.create(FlagGrid)
vel      = s.create(MACGrid)
velDivFree = s.create(MACGrid)
density  = s.create(RealGrid)
pressure = s.create(RealGrid)
pressure2 = s.create(RealGrid)

# noise field for more interesting source
noise = s.create(NoiseField, loadFromFile=True)
noise.posScale = vec3(res)
noise.clamp = True
noise.clampNeg = 0.9
noise.clampPos = 1
noise.valOffset = 9.5
noise.valScale = 0.1
noise.timeAnim = 0.02

source = s.create(Sphere, center=float(res)*vec3(0.5,0.1,0.5), radius=res*0.05)

bWidth=1
flags.initDomain( boundaryWidth=bWidth )
flags.initDomain()
flags.fillGrid()

setOpenBound( flags, bWidth,'xXYzZ',FlagOutflow|FlagEmpty )

if (GUI):
    gui = Gui()
    gui.show()

outdir = "out/plume_" + advectorName + ("_" + methodName if methodName else "") + "_" + str(res)
try:
    os.makedirs(outdir)
except Exception as e:
    print(str(e))
    print("Could not create output folder")
    sys.exit(1)

buoyancyGravity = -6e-4 #gets scaled in addBuoyancy by inverse of dx (where dx = 1.0 / mGridSize.max())

def step():
    densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)

    density.save('%s/density_%04d.vol' % (outdir, s.frame))
    velDivFree.save('%s/velocityGrid_%04d.uni' % (outdir, s.frame))
    
    advectSemiLagrange(flags=flags, vel=velDivFree, grid=density, order=2, orderSpace=2)
    advectSemiLagrange(flags=flags, vel=velDivFree, grid=vel, order=advectionOrder, orderSpace=1)
    
    resetOutflow( flags=flags, real=density )
    
    addBuoyancy(density=density, vel=vel, gravity=vec3(0,buoyancyGravity,0), flags=flags)
    setWallBcs(flags=flags, vel=vel)

def project():
    p = pressure2 if newMethod else pressure
    solvePressure( flags=flags, vel=vel, pressure=p )

def reflect():
    velDivFree.copyFrom(vel)
    solvePressure( flags=flags, vel=velDivFree, pressure=pressure )
    # reflect vel about velDivFree
    vel.multConst(vec3(-1,-1,-1))
    vel.addScaled(velDivFree,vec3(2,2,2))

#main loop
while s.frame < 500:
    mantaMsg('\nFrame %i' % (s.frame))
    
    velDivFree.copyFrom(vel)
    if newMethod:
        step()
        reflect()
        s.step()
        step()
        project()
        s.step()
    else:
        step()
        project()
        s.step()

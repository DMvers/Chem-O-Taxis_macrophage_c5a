#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:39:00 2024
@author: dmv
"""
#This is a simulation of macrophages crossing the bridge of an Insall chamber

#these are libraries from the virtual environment
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numba
import math
import argparse

#these are supporting files in the local folder
import mazelayouts
import environment
import cell
import collisionfunctions
import datasaver


#initialise our arguments
parser = argparse.ArgumentParser()
#General
parser.add_argument("-folder",default="testbed", help="Where to save the output")
parser.add_argument("-plotting",default = 500,help="How often to make a plot, 0 = never 1 = every timestep 2 = every two timesteps etc.",type=int)
parser.add_argument("-saving",default = 50,help="How often to save data",type=int)

#Environment
parser.add_argument("-Nx",default=200, help="Width of the grid",type=int)
parser.add_argument("-Ny",default=210, help="Height of the grid",type=int)
parser.add_argument("-steps",default=28800, help="Duration of the simulation",type=int)
parser.add_argument("-cells",default=150, help="Maximum number of cells to place, default is 100",type=int)
parser.add_argument("-gradient",default=1, help="What type of initial gradient to use for the attractant, 0 for no gradient",type=int)
parser.add_argument("-highgradientstrength",default=5.00E-08, help="What strength the highest part of the gradient should have",type=float)
parser.add_argument("-initialattractant",default=5.00E-09, help="How much attractant to place as a minimum. used as low end of gradient if gradient is on",type=float)
parser.add_argument("-attractantplacementarea",default=3, help="What area to use to continue placing new attractant while running the model",type=int)


#General cell properties
parser.add_argument("-cellsize",default=3, help="Size of cells in grid site equivalents",type=float)
parser.add_argument("-mitogenfactor",default=0, help="How much more to grow with attractant",type=float)
parser.add_argument("-basemitosis",default=0, help="How often to try to divide",type=int)
parser.add_argument("-basedeath",default=0, help="How often to die",type=float)

#Cell movement properties
parser.add_argument("-collision",default=1, help="Whether to have collision detection. 0 = only for walls, 1= distance-based for cells, 2=shape-based for cells (would allow for complex shapes)",type=int)
parser.add_argument("-celldistancefactor",default=2, help="How much distance to keep in cell sizes if collision is on",type=float)
parser.add_argument("-persistence",default=0.85, help="How much cells want to continue their course (0-1)",type=float)
parser.add_argument("-movedistance",default=0.008, help="How far should cells move per step",type=float)

#Chemotaxis cell properties

parser.add_argument("-cellfuzzingrelative",default=0.2, help="How much fuzzier binding values should be made relative to calculated binding (0-~1)",type=float)
parser.add_argument("-cellfuzzingabsolute",default=0, help="How much extra noise should be added to binding values(0-~1)",type=float)
parser.add_argument("-envirofuzzingabsolute",default=2E-09, help="How much extra noise should be added to sensed attractant values per attractant(0-~1)",type=float)


parser.add_argument("-attractantweighing",default=1, help="How strongly to sense the attractant",type=float)

parser.add_argument("-attractantconsumption1kd",default=1.00E-09, help="kd for consumption",type=float)
parser.add_argument("-attractantconsumption1vmax",default=6.59E-11, help="vmax for consumption",type=float)

parser.add_argument("-attractantconsumption2kd",default=4.00E-07, help="kd for consumption",type=float)
parser.add_argument("-attractantconsumption2vmax",default=0 , help="vmax for consumption",type=float)

parser.add_argument("-attractantreceptorkd",default=1.00E-09, help="What kd to use for sensing. If 0, default to directly reading the concentration",type=float)
parser.add_argument("-attractantdiffusion",default=12, help="Diffusion of attractant",type=float)


#Load arguments
args = parser.parse_args()

#initialise our datastorage
mydatasaver = datasaver.datasaver(args.folder)
plotinterval = args.plotting
saveinterval = args.saving
# Environment parameters
Nx = args.Nx
Ny = args.Ny
steps = args.steps
maxcells = args.cells 
gradienttype = args.gradient
attractantplacementarea = args.attractantplacementarea
highgradient = args.highgradientstrength

#Cell properties
cellsize = args.cellsize
mitogenfactor = args.mitogenfactor
basemitosis = args.basemitosis
basedeath = args.basedeath
attractantreceptorkd = args.attractantreceptorkd

#Cell movement properties
cellcollision = args.collision
celldistance = cellsize*args.celldistancefactor
persistence = args.persistence
movedistance= args.movedistance
cellfuzzingrelative = args.cellfuzzingrelative
cellfuzzingabsolute = args.cellfuzzingabsolute
envirofuzzingabsolute = args.envirofuzzingabsolute

#parameters for ligands
#attractantparameters - this represents a primary attractant
initialattractant = args.initialattractant
attractantdiffusion = args.attractantdiffusion #How fast it diffuses - a high number means it diffuses a lot
attractantweighing = float(args.attractantweighing)#How strongly this is sensed

#breakdown 1
attractantconsumption1kd = args.attractantconsumption1kd
attractantconsumption1vmax = args.attractantconsumption1vmax

#breakdown 2
attractantconsumption2kd = args.attractantconsumption2kd
attractantconsumption2vmax = args.attractantconsumption2vmax


#Set number of diffusion steps
diffusionrepeats=1
maxdiffuse = 0.2
if attractantdiffusion>maxdiffuse:
    diffusionrepeats = math.ceil(attractantdiffusion/maxdiffuse)
    attractantdiffusion= attractantdiffusion/diffusionrepeats        
        
# Create the grid walls
grid_walls = mazelayouts.makeemptydish(Nx, Ny)
alloccupiedsites = np.array([])
wallsites = np.array([])

#Add walls to the set of occupied sites
for i in range(0,Nx):
    for j in range(0,Ny):
        if grid_walls[i,j]>0:
            alloccupiedsites = np.union1d(alloccupiedsites,j*Nx+i)
            wallsites = np.union1d(wallsites,j*Nx+i)


#Create the chemical grids
ligands = {}
ligands["attractant"] = environment.ligand(Nx,Ny,0,attractantdiffusion,grid_walls,"attractant")

#A single value to represent the concentration in the outer well
modelsquares = 20000
outerwellvalue = (initialattractant/1000)*modelsquares/0.00000001#initialattractant
correctedouterwellvalue = outerwellvalue*8.00E-14*1000

#Fill part of the grid with attractant
for i in range(1, ligands["attractant"].grid.shape[0]-1):
     for j in range(1, ligands["attractant"].grid.shape[1]-1):
         if grid_walls[i,j]<1:
             if gradienttype ==0:
                ligands["attractant"].grid_prev[i,j] = initialattractant
             if gradienttype ==1:
                ligands["attractant"].grid_prev[i,j] = initialattractant +(highgradient -initialattractant)*((Ny-j)/Ny)
             if gradienttype ==2:
                if j<0.2*Ny:
                    ligands["attractant"].grid_prev[i,j] = 2*initialattractant
             if gradienttype ==3:
                 ligands["attractant"].grid_prev[i,j] = ((((Ny-j)/Ny)+0.5)/20)*initialattractant
        
#Do a first diffusion step on the grid, this is neccesary to do dufort diffusion later
for ligand in ligands.values():
    ligand.diffuse_euler_init(grid_walls)

#Place the cells
cells = list()

#pre-allocate space is we're saving their central locations only
if cellcollision ==1:
    cellxlocations = np.zeros(maxcells*10) #Make a guess for how much space we'll need
    cellylocations = np.zeros(maxcells*10)
      
for i in range(0,maxcells):
    unplaced = True
    attempt = 0
    while unplaced:
        #test if new cell would not overlap with any other cells or walls
        xlocation =random.randint(int(0.03*Nx)+cellsize, int(0.97*Nx)-cellsize)
        ylocation =random.randint(Ny-50,Ny-6)
        newcell = cell.cell(xlocation,ylocation,cellsize,cellcollision,cellfuzzingrelative,cellfuzzingabsolute,Nx,Ny,i)
        if cellcollision == 0:
            cells.append(newcell)
            newcell.definesurfacesquares()
            break
        
        if cellcollision == 1:
            if collisionfunctions.check_collision_distance(i,xlocation,ylocation,cellxlocations,cellylocations,celldistance):
                cellxlocations[i] = xlocation
                cellylocations[i] = ylocation
                cells.append(newcell)
                newcell.definesurfacesquares()
                break
            
        if cellcollision == 2:
            newoccupiedsites = newcell.definesurfacesquares()
            if len(np.intersect1d(alloccupiedsites,newoccupiedsites))==0: #This means we've placed it at an available stop
                alloccupiedsites = np.union1d(alloccupiedsites,newcell.definesurfacesquares())
                cells.append(newcell)
                break
            
        #This is only triggered if cellcollision was >0 and we failed to place the cell
        attempt +=1
        if attempt > 100:
            del newcell
            print("Unable to place cell!")
            print("placed cells")
            print(i)
            break

#Increment the counter with one, so that any new cells that are created will get a new number            
maxcells = maxcells + 1

##Run the main loop of our simulation
for t in range(steps):
    start = time.time()
    
    
    for i in range(1, ligands["attractant"].grid.shape[0]-1):
        for j in range(1, ligands["attractant"].grid.shape[1]-1):
            if j<(attractantplacementarea):
                ligands["attractant"].grid_prev[i,j] = initialattractant
                ligands["attractant"].grid[i,j] = initialattractant
            if gradienttype == 1:
                if j>(Ny-3):
                    ligands["attractant"].grid_prev[i,j] = 0
                    ligands["attractant"].grid[i,j] = 0
    
    startdiffuse = time.time()
    #Diffuse ligands
    shape = ligands["attractant"].grid.shape
    for ligand in ligands.values():
        diffusionrepeated = 1
        ligand.diffuse_euler(grid_walls)
        while diffusionrepeated <diffusionrepeats:
            if attractantplacementarea > 1:
                end_col = min(attractantplacementarea, shape[1] - 1)
                ligands["attractant"].grid_prev[1:-1, 1:end_col] = initialattractant+highgradient
                ligands["attractant"].grid[1:-1, 1:end_col] = initialattractant+highgradient

            ligands["attractant"].grid_prev[1:-1, (Ny-5):-1] = correctedouterwellvalue
            ligands["attractant"].grid[1:-1, (Ny-5):-1] = correctedouterwellvalue
            ligand.diffuse_dufort(grid_walls)
            diffusionrepeated+=1
    enddiffuse = time.time() 
    
    #Save data every this many steps using feather
    if(t%saveinterval==0):
        attractantname = "attractant"
        metabolitename = "metabolite"
        cellname = "cells"
        startsave = time.time()
        mydatasaver.savegrid(ligands["attractant"].grid, attractantname+ str(t))
        mydatasaver.savecells(cells, cellname+ str(t))
        endsave = time.time()
    
    mydatasaver.savecellscsv(cells,"tracking",t)
    #Plot the current state every so many steps
    if plotinterval >0:
        if(t%plotinterval==0):

            startplot= time.time()
            plt.clf()
            plt.imshow(ligands["attractant"].grid.T, origin='lower', extent=[0, ligands["attractant"].grid.shape[0], 0,ligands["attractant"].grid.shape[1]], cmap='viridis', interpolation='none',vmin=0,vmax=initialattractant+highgradient)


            plt.title("Time = "+str(t)+" steps")
            for thiscell in cells:
                cellcircle = patches.Circle((thiscell.xlocation, thiscell.ylocation), thiscell.size, linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(cellcircle)
            plt.show()
            endplot = time.time()
            print("plottime (not included)")
            print(endplot-startplot)
            
    #Create fuzzed ligands if neccesary
    if envirofuzzingabsolute>0:
        fuzzedligands = []
        for ligand in ligands.values():
            fuzzedligands.append(ligand.grid+np.random.uniform(0,envirofuzzingabsolute,size=ligand.grid.shape))
    
    startmove = time.time()
    #shuffle the list so we adress them in a random order
    random.shuffle(cells)
    #Have cells sense chemoattractant, set their target accordingly, consume and produce ligands, and move
    for thiscell in cells:
        #sensing
        if envirofuzzingabsolute>0:
            thiscell.sense_multiple_attractants_fuzzedgrids(fuzzedligands,[attractantweighing],persistence,[[0]],[attractantreceptorkd])
        else:
            thiscell.sense_multiple_attractants(ligands,[attractantweighing],persistence,[[0]],[attractantreceptorkd])
  
        #consuming
        totalconsumed = 0
        if attractantconsumption1vmax >0:
            consumedpersite = ligands["attractant"].consume(thiscell.occupiedsites, attractantconsumption1kd,attractantconsumption1vmax,returndetail=False)
        if attractantconsumption2vmax >0:
            consumedpersite = ligands["attractant"].consume(thiscell.occupiedsites, attractantconsumption2kd,attractantconsumption2vmax,returndetail=False)
           
        #collision checking
        if thiscell.collision ==0:
            thiscell.move_simple(movedistance,grid_walls)
        if thiscell.collision ==1:
            newxloc,newyloc = thiscell.move_coarse(movedistance,Nx,Ny,grid_walls,cellxlocations,cellylocations,celldistance)
            if ((newxloc == cellxlocations[thiscell.id]) and (newyloc == cellylocations[thiscell.id])):
                thiscell.targetangle = random.uniform(-1*math.pi,math.pi)
            else:
                cellxlocations[thiscell.id], cellylocations[thiscell.id] = newxloc,newyloc
             
        if thiscell.collision ==2:
            alloccupiedsites = thiscell.move_fine(movedistance,Nx,Ny,alloccupiedsites)
            
  
    
    #check if we need to add more space to the cellocationarray
    if cellcollision == 1:
        if (maxcells*2) > (len(cellxlocations)):
            cellxlocations = np.append(cellxlocations,np.zeros(len(cellxlocations)))
            cellylocations = np.append(cellylocations,np.zeros(len(cellxlocations)))
    
    #Change the concentration in the outer well if relevant
    if initialattractant>0:
        totaloutercells = 1000000 # per ml
        outerwellsquareequivalent = 2.00E+14
        volumepersquare = 5.00E-13
        attractant1vmaxouterwell = (attractantconsumption1vmax*28/1000*totaloutercells*10000) # picomol per cell site to umol per Liter
        cons1kd = attractantconsumption1kd/1000/volumepersquare #Go to umol per liter from picomol per grid site
        outerwellvalue = outerwellvalue- ((outerwellvalue*attractant1vmaxouterwell)/(outerwellvalue+cons1kd)) 
        correctedouterwellvalue = outerwellvalue*volumepersquare*1000
        print(correctedouterwellvalue)
    else:
        correctedouterwellvalue = 0
    #timekeeping
    print("step")
    print(t)
    endmove = time.time()
    
    end = time.time()
    
# Final plot after the loop
plt.show()

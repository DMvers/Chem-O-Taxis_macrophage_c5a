library(arrow)
library(colorspace)

source("support.r")

filenamea = "attractant"
filenamec = "cells"
interval=500
startnumber = 0
endnumber = 21600
currentfolder = getwd()

runs = c("")



    allexpnames = c("uniform0","uniform03","uniform3","uniform10","uniform30","uniform100","gradient10")
for(foldername in allexpnames){
runname = "run1"
fullname = paste(foldername,runname,sep="")
setwd(paste(currentfolder,"/",fullname,sep=""))
visualiseonerunoneattractant(filenamea,filenamec,interval,0,endnumber,paste(foldername,runname,sep=""))
setwd(currentfolder)
    }


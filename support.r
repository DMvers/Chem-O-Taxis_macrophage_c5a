visualiseonerunoneattractant<-function(filenamea,filenamec,interval,startnumber,endnumber,experimentname){
    initialdistance = 0
framenumber = 0
allcelllocations = list()
allxlocations = list()
allylocations = list()
allprevxlocations = list()
allprevylocations = list()

first_grad = create_gradient(0.3, 1, 0,a=0.5)   # green gradient
second_grad = create_gradient(0, 0, 1,a=0.5)  # blue gradient

    for(i in seq(startnumber,endnumber,by=interval)){
    framenumber = framenumber +1
    paddedi = sprintf("%04d", framenumber) #This will be used for filenames later
    print(paddedi)
    #Load grid
    grid = read_feather(paste(filenamea,i,sep=""), col_select = NULL, as_data_frame = TRUE, mmap = TRUE)
    ylength = 210
    xlength = 200
    displayable = matrix(unlist(grid),nrow=ylength,ncol=xlength)
    print(mean(displayable))

    #Load cells
    cells = read_feather(paste(filenamec,i,sep=""), col_select = NULL, as_data_frame = TRUE, mmap = TRUE)
    cellx = cells[,1]
    celly = cells[,2]
    cellid = cells[,3]+1#can't have ids be 0 in R 
    for (j in 1:length(cellid)){
        if(i==startnumber){
            allcelllocations[[cellid[j]]]=list(list(cellx[j],celly[j]))
            allxlocations[[cellid[j]]] = c(cellx[j])
            allylocations[[cellid[j]]] = c(celly[j])
            allprevxlocations[[cellid[j]]] = c(cellx[j]) #This makes the current and previous be of equal length, at the cost of drawing one 0-length line segment
            allprevylocations[[cellid[j]]] = c(celly[j])
        }
        else{
            allcelllocations[[cellid[j]]] = append(allcelllocations[[cellid[j]]],list(list(cellx[j],celly[j])))
            allprevxlocations[[cellid[j]]] = c(allprevxlocations[[cellid[j]]],tail(allxlocations[[cellid[j]]],1))
            allprevylocations[[cellid[j]]] = c(allprevylocations[[cellid[j]]],tail(allylocations[[cellid[j]]],1))
            allxlocations[[cellid[j]]] = c(allxlocations[[cellid[j]]],cellx[j])
            allylocations[[cellid[j]]] = c(allylocations[[cellid[j]]],celly[j])
            
        }
    }
     maketrackinggridoneattractant(displayable,segments,cellid,celly,ylength,cellx,xlength,allxlocations,allylocations,allprevxlocations,allprevylocations,first_grad,second_grad,experimentname,paddedi)

    }
}

maketrackinggridoneattractant   <-function(displayable,segments,cellid,celly,ylength,cellx,xlength,allxlocations,allylocations,allprevxlocations,allprevylocations,colour1,colour2,experimentname,paddedi){
       
print(experimentname)
png(paste(experimentname,"gridtracked",paddedi,".png",sep=""),width=1000,height=400)
    par(mar = c(0,0,0,0),xaxs="i",yaxs="i")
    plot.new()

    #print(mean(displayable))
    displayable = displayable*1E10
    high = max(displayable)
    image(displayable,col = colour1,zlim=c(0,high),axes=FALSE,breaks = seq(0, high, length.out = length(colour1) + 1),useRaster=TRUE)
    
    for(j in 1:length(cellid)){
        tracklength = length(allprevylocations[[j]])
        visibletracklength = 50
        if(tracklength>visibletracklength){
            fadevector = c(rep(0,length(allprevylocations[[j]])-visibletracklength),seq(from = 0, to = 1,length.out = visibletracklength))
        }
        else{
             fadevector = seq(from = 0, to = 1,length.out = tracklength)
        }
        
        segments(allprevylocations[[j]]/ylength,allprevxlocations[[j]]/xlength,allylocations[[j]]/ylength,allxlocations[[j]]/xlength,col=rgb(0.7,0.7,0.7,fadevector),lwd=4) #This draws the line segments on the grid

    }
    points(celly/ylength,cellx/xlength,col="red1",cex=5,pch=19) #Yes, this is mirrored, the image automirrors, so this matches
    hours = floor((length(allxlocations[[1]])*1500)/60/60)
    text(labels=(paste(hours,"h",sep="")), x=0.9,y=0.1,cex=3,col="white",srt=270)

    dev.off()#Finish making our grid image

}

create_gradient <- function(r, g, b, n = 100,a=0.5) {
  rgb(
    red = r*seq(0, 1, length.out = n),
    green = g*seq(0, 1, length.out = n),
    blue = b*seq(0, 1, length.out = n),
    alpha = a # Opaque -> Transparent
  )
}

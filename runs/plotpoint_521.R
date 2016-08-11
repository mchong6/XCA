# Classical MDS
# N rows (objects) x p columns (variables)
# each row identified by a unique row name

library(rgl)

directory = "CSV_521"
#concatenate the directory to path
XA = as.matrix(read.csv(paste(directory,"/exclusiveA_outputs.csv", sep = '', collapse = '')))
XB = as.matrix(read.csv(paste(directory,"/exclusiveB_outputs.csv", sep = '', collapse = '')))
common = as.matrix(read.csv(paste(directory,"/common_outputs.csv", sep = '', collapse = '')))

XA_label = t(matrix(c(1,0,0,0,0)))
XB_label = t(matrix(c(0,0,1,0,0)))
common_label = t(matrix(c(0,0,0,0,1)))

set = rbind(XA_label,XB_label,common_label, XA, XB, common)

#this just names the points so we can group up the same names as a single color
labels_XA = as.matrix(rep(1, 1))
labels_XB = as.matrix(rep(2, 1))
labels_common = as.matrix(rep(3, 1))
labels1 = as.matrix(rep(4, 99))
labels2 = as.matrix(rep(5, 99))
labels3 = as.matrix(rep(6, 99))

labels = rbind(labels_XA,labels_XB,labels_common,labels1,labels2,labels3)

row.names(set) <- labels
family <- as.factor(row.names(set))
#col = family
#col.rainbow <- rainbow(12)
#col.topo <- topo.colors(12)
#col.terrain <- terrain.colors(12)
#palette(col.rainbow)

d <- dist(set) # euclidean distances between the rows
fit <- cmdscale(d,eig=TRUE, k=3) # k is the number of dim
#fit # view results

# plot solution
x <- fit$points[,1]
y <- fit$points[,2]
z <- fit$points[,3]

#scatterplot3d(x,y,z, color = family, cex.symbols = .5, angle = 350)
s3d = plot3d(x,y,z, col = rainbow(6)[family], type = 's', size = .8)
legend3d("topright", legend = paste('Type', c('CorXA','CorXB','CorC','XA','XB','Common')), pch = 16, col = rainbow(6), inset=c(0.02))
#writeWebGL(dir=paste("cifar.torch/",directory, sep = '', collapse = ''), "webGL")
paste("file://", writeWebGL(directory, width=700), sep="")

#plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", main="Metric MDS", col = family, pch = 0, cex = .5)
#text(x, y, labels = row.names(set), cex=.7) 
#text(x, y, labels = row.names(set)[1:499], cex=.4, pch = 0,col= family) 
#text(x, y, labels = row.names(set)[500:999], cex=.4, col="blue") 
#text(x, y, labels = row.names(set)[1000:1998], cex=.4, col="green")
#text(x, y, labels = row.names(set)[1999:2497], cex=.4, col="blueviolet")
#text(x, y, labels = row.names(set)[2498:2997], cex=.4, col="dimgrey")

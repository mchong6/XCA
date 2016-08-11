# Classical MDS
# N rows (objects) x p columns (variables)
# each row identified by a unique row name

library(rgl)

directory = "CSV_521"
#concatenate the directory to path
XA = as.matrix(read.csv(paste(directory,"/visualize_exclusiveA.csv", sep = '', collapse = '')))
XB = as.matrix(read.csv(paste(directory,"/visualize_exclusiveB.csv", sep = '', collapse = '')))
common = as.matrix(read.csv(paste(directory,"/visualize_common.csv", sep = '', collapse = '')))
set = rbind(XA, XB, common)

labels1 = as.matrix(rep(1, 999))
labels2 = as.matrix(rep(2, 1000))
labels3 = as.matrix(rep(3, 1999))
labels4 = as.matrix(rep(4, 999))
labels5 = as.matrix(rep(5, 1000))
labels = rbind(labels1,labels2,labels3,labels4,labels5)

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
s3d = plot3d(x,y,z, col = rainbow(5)[family])
legend3d("topright", legend = paste('Type', c('A,A', 'B,A', 'AB,C', 'A,B', 'B,B')), pch = 16, col = rainbow(5), inset=c(0.02))
#writeWebGL(dir=paste("cifar.torch/",directory, sep = '', collapse = ''), "webGL")
paste("file://", writeWebGL(directory, width=700), sep="")

#plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", main="Metric MDS", col = family, pch = 0, cex = .5)
#text(x, y, labels = row.names(set), cex=.7) 
#text(x, y, labels = row.names(set)[1:499], cex=.4, pch = 0,col= family) 
#text(x, y, labels = row.names(set)[500:999], cex=.4, col="blue") 
#text(x, y, labels = row.names(set)[1000:1998], cex=.4, col="green")
#text(x, y, labels = row.names(set)[1999:2497], cex=.4, col="blueviolet")
#text(x, y, labels = row.names(set)[2498:2997], cex=.4, col="dimgrey")

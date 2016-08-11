# Classical MDS
# N rows (objects) x p columns (variables)
# each row identified by a unique row name

library(rgl)

directory = "CSV_541"
#concatenate the directory to path
#XA is the points in space of A and B through filter A
XA = as.matrix(read.csv(paste(directory,"/visualize_exclusiveA.csv", sep = '', collapse = '')))
XB = as.matrix(read.csv(paste(directory,"/visualize_exclusiveB.csv", sep = '', collapse = '')))
common = as.matrix(read.csv(paste(directory,"/visualize_common.csv", sep = '', collapse = '')))
set = rbind(XA, XB, common)

#importing csv cause the first value to disappear thus the 999 instead of 1000 for the first group.
#im naming the different points
labels1 = as.matrix(rep(1, 999)) #image A -> filter A red color
labels2 = as.matrix(rep(2, 1000))#image B -> filter B yellow color
labels3 = as.matrix(rep(3, 1999))#green color
labels4 = as.matrix(rep(4, 999))#blue
labels5 = as.matrix(rep(5, 1000))#purple
#rbind is like concat
labels = rbind(labels1,labels2,labels3,labels4,labels5)

#naming the points in set
row.names(set) <- labels
#something to do with color
family <- as.factor(row.names(set))

#col = family
#col.rainbow <- rainbow(12)
#col.topo <- topo.colors(12)
#col.terrain <- terrain.colors(12)
#palette(col.rainbow)

d <- dist(set) # euclidean distances between the rows
fit <- cmdscale(d,eig=TRUE, k=3) # k is the number of dimensions
#fit # view results

# plot solution
x <- fit$points[,1]
y <- fit$points[,2]
z <- fit$points[,3]


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

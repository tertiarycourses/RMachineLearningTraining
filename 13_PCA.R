###########################################################################
# Useful for finding low dimensional representation of the correlated data
###########################################################################
cor(iris[,1:4])

names(iris)
plot(iris$Sepal.Length, col = iris$Species)
plot(iris$Sepal.Width, col = iris$Species)

pc1 <- prcomp(iris[,3:4], scale = TRUE)
plot(pc1$x[,1], col = iris$Species)

pc1$rotation

iris1 <- scale(iris[,3:4])
iris1[1:5,]%*%pc1$rotation
pc1$x[1:5,]

## PCA on full data set
pc2 <- prcomp(iris[,1:4], scale = TRUE)
pc2$x[1:3,]

plot(pc2$x[,1], col = iris$Species)

iris2 <- scale(iris[,1:4])
iris2[1:5,]%*%pc2$rotation
pc2$x[1:5,]


summary(pc2)
vars <- apply(pc2$x, 2, var)  
props <- vars / sum(vars)
cumsum(props)

barplot(cumsum(props))


pc2$rotation

## % variance explained by  principle components
pc2$sdev
pc2V <- pc2$sdev^2
pve <- pc2V/sum(pc2V)
pve
plot(pve, xlab = "Principal Component", ylab = "Proportion of 
     variance explained", ylim=c(0,1), type="b")

plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of 
     variance explained", ylim=c(0,1), type="b")


### visualization

library(rgl)

col1 <- c("red","green","blue")
plot3d(pc2$x[,1],pc2$x[,2],pc2$x[,3], size=6, col=col1, xlab="", ylab="", zlab="", sub="3 clusters")
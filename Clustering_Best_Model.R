############################################ CLUSTERING ##################################################
library(cluster)
library(ggplot2)
library (NbClust)
library(factoextra)
set.seed(123)
load("~/Downloads/cluster_data.RData")

y_scaled <- scale(y, center = TRUE, scale = TRUE)        #scale and mean center the data 

pca <- prcomp(y_scaled, center = TRUE, scale. = TRUE)
variance <- pca$sdev^2 / sum(pca$sdev^2)
cumulative_variance <- cumsum(variance)
num_dims <- which(cumulative_variance >= 0.95)[1]
y_pca <- data.frame(pca$x[, 1:num_dims])

###########find optimal k using the nbclust function for silhoutte method##########
fviz_nbclust(y_pca, kmeans, method = "silhouette")


##################Perform k-means clustering for evaluated value of k####################
set.seed(123)
final <- kmeans(y_pca,2 , nstart = 25)           
print(final)

fviz_cluster(final, data = y_pca,geom = "point")
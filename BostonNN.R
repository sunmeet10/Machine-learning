#Boston Housing Price Prediction using Neural Network

#install.packages("neuralnet")
library(Metrics)
library(MASS)
data <- Boston

# Set a seed
set.seed(1000)

# Check that no data is missing
apply(data,2,function(x) sum(is.na(x)))

# Train-test random splitting 
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]

# Neural net fitting

# Scaling data for the NN
maxs <- apply(data, 2, max)
maxs
mins <- apply(data, 2, min)
mins
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
head(scaled)

# Train-test split
train_set <- scaled[index,]
test_set <- scaled[-index,]

# NN training
library(neuralnet)
n <- names(train_set)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_set,hidden=c(5,3),linear.output=T, act.fct = 'logistic')

# Predict
pr.nn <- compute(nn,test_set[,1:13])

# Results from NN are normalized (scaled)
# Descaling for comparison
pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

# Calculating MSE
MSE.nn<-mse(test.r, pr.nn_)
MSE.nn

# Improve Performance - Change Parameter
nn_new <- neuralnet(f,data=train_set,hidden=c(10,5,3),linear.output=T, act.fct = 'logistic')
pr.nn_new <- compute(nn_new,test_set[,1:13])
pr.nn_new <- pr.nn_new$net.result*(max(data$medv)-min(data$medv))+min(data$medv)

# Re-calculate MSE
MSE.nn_new<-mse(test.r, pr.nn_new)
MSE.nn_new

# Compare Performance
print(paste('Old MSE :',MSE.nn, 'New MSE :', MSE.nn_new))

# Plot predictions - Old MSE
plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7, xlab = "Actual", ylab = "Predicted")
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')


# Plot predictions - New MSE
plot(test$medv,pr.nn_new,col='red',main='Real vs predicted NN',pch=18,cex=0.7, xlab = "Actual", ylab = "Predicted")
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

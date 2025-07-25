
library(class)          
library(gmodels)     
library(ggplot2)      
library(rpart)        
library(rpart.plot)   
library(neuralnet)    
library(e1071)  
library(arules)
library(cluster)


data <- diamonds

data <- data[, -c(1, 3, 4)]

table(data$cut)
colors <- rainbow(length(unique(data$cut)))

# Plot with the colorful bars
bar_plot <- plot(data$cut, 
                 main = "Bar Plot of Cut", 
                 col = colors,  # Apply the rainbow colors
                 xlab = "Cut", 
                 ylab = "Frequency")


hist(data$price)

round(prop.table(table(data$cut)) * 100, digits = 2)

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
data2 <- as.data.frame(lapply(data[, 2:6], normalize))  # Corrected
# Split data into training and testing sets
set.seed(123)  # For reproducibility
index <- sample(1:nrow(data2), 0.8 * nrow(data2))
data_train <- data2[index, ]
data_test <- data2[-index, ]

# Extract training and testing labels for the target variable `cut`
data_trainLa <- data$cut[index]
data_testLa <- data$cut[-index]

# --- k-NN classification ---
pred_knn <- knn(train = data_train, test = data_test, cl = data_trainLa, k = 19)
# Evaluate predictions using CrossTable
CrossTable(x = pred_knn, y = data_testLa, prop.chisq = FALSE)
# Confusion matrix and accuracy
confusion_knn <- table(pred_knn, data_testLa)
cat("k-NN Confusion Matrix:\n")
print(confusion_knn)
acc_knn <- sum(diag(confusion_knn)) / sum(confusion_knn)
cat("k-NN Accuracy:", round(acc_knn * 100, 2), "%\n")

# --- Naive Bayes classification ---
# Convert target variable to factor
data_trainLa <- factor(data_trainLa)
nb_model <- naiveBayes(data_train, data_trainLa)
nb_pred <- predict(nb_model, newdata = data_test)
# Evaluate predictions using CrossTable
cat("\nNaive Bayes Classification Results:\n")
CrossTable(x = nb_pred, y = data_testLa, prop.chisq = FALSE)
# Confusion matrix and accuracy
nb_confusion <- table(nb_pred, data_testLa)
cat("Naive Bayes Confusion Matrix:\n")
print(nb_confusion)
# Accuracy
nb_acc <- sum(diag(nb_confusion)) / sum(nb_confusion)
cat("Naive Bayes Accuracy:", round(nb_acc * 100, 2), "%\n")

# --- Decision Tree classification ---
data2$cut <- data$cut
dt_model <- rpart(cut ~ ., data = data2, method = "class")

# Make predictions
dt_pred <- predict(dt_model, newdata = data2, type = "class")

# Evaluate predictions using CrossTable
cat("\nDecision Tree Classification Results:\n")
CrossTable(x = dt_pred, y = data$cut, prop.chisq = FALSE)

# Confusion matrix and accuracy
dt_confusion <- table(dt_pred, data$cut)
cat("Decision Tree Confusion Matrix:\n")
print(dt_confusion)

# Accuracy
dt_acc <- sum(diag(dt_confusion)) / sum(dt_confusion)
cat("Decision Tree Accuracy:", round(dt_acc * 100, 2), "%\n")

# Plot the decision tree using rpart.plot
rpart.plot(dt_model, main = "Decision Tree for Diamond Cut Prediction")



# --- Neural Network classification ---

library(nnet)

# Convert the target variable to a factor for classification
data_trainLa <- as.factor(data_trainLa)
data_testLa <- as.factor(data_testLa)

# Train the neural network
set.seed(123)  # For reproducibility

nn_model <- nnet(x = data_train, y = class.ind(data_trainLa),
                 size = 5, # Number of hidden neurons (tune as needed)
                 maxit = 200, # Maximum number of iterations
                 decay = 0.01, # Weight decay for regularization
                 softmax = TRUE) # Softmax for multi-class classification



# Make predictions on the test set
nn_predictions <- predict(nn_model, data_test, type = "class")
confusion_matrix <- table(Predicted = nn_predictions, Actual = data_testLa)
print("Confusion Matrix:")
print(confusion_matrix)
CrossTable(nn_predictions,y=data_testLa)
# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))


# k means cluster

data<-data[-1]
kmeans_result <- kmeans(data, centers = 3, nstart = 25)
print(kmeans_result)

kmeans_result$cluster        #cluster assignment for each data type 
kmeans_result$centers      #contains the centroids of the clusters.

cm <- table(kmeans_result$cluster)
cm

plot(data2[, c("price", "depth")],
     col = kmeans_result$cluster,
     main = "K-means with 3 clusters",
     xlab = "price",
     ylab = "depth")

points(kmeans_result$centers[, c("price","depth")],
       col = 1:3, pch = 8, cex = 3)


y_kmeans <- kmeans_result$cluster

cluster_data <- data.frame(Cluster = kmeans_result$cluster, TrueLabel = data_testLa)

cluster_table <- table(cluster_data$Cluster, cluster_data$TrueLabel)

cluster_accuracy <- apply(cluster_table, 1, function(x) max(x) / sum(x))

clustering_accuracy <- sum(cluster_accuracy * table(kmeans_result$cluster) / length(kmeans_result$cluster))

# Print the clustering accuracy
print(paste("Clustering Accuracy:", round(clustering_accuracy * 100, 2), "%"))


# Store the accuracy of each model
acc_values <- c(
  "k-NN" = acc_knn,              # k-NN accuracy
  "Naive Bayes" = nb_acc,        # Naive Bayes accuracy
  "Decision Tree" = dt_acc,      # Decision Tree accuracy
  "Neural Network" = accuracy,   # Neural Network accuracy
  "Clustering" = clustering_accuracy # Clustering accuracy
)

# Create a rainbow color palette based on the number of accuracy values
rainbow_colors <- rainbow(length(acc_values))

# Create a bar plot of the accuracy values with the rainbow color gradient
barplot_heights <- barplot(
  acc_values,
  main = "Model Accuracy Comparison",
  col = rainbow_colors,          # Apply the rainbow gradient
  ylab = "Accuracy (%)",
  xlab = "Models",
  border = "black",              # Bar border color
  ylim = c(0, 1),                # Limiting y-axis to 0-1 (for 0% to 100%)
  space = 0.25                   # Adjust space between bars (75% width for each bar)
)

# Add accuracy labels on top of each bar
text(
  x = barplot_heights,
  y = acc_values,                # Position the text at the height of the bars
  labels = round(acc_values * 100, 2),  # Round accuracy values to 2 decimal places
  pos = 3,                       # Position the text slightly above the bars
  col = "black"                  # Color of the labels
)


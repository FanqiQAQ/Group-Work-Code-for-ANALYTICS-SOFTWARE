library(xgboost)
library(caret)
library(dplyr)
library(parallel)
library(pROC)
library(PRROC)
library(ggplot2)
install.packages("readr")
library(readr)
# 1: load data
#----------------------------------------------------------------
setwd("C:/Users/mengfanqi/Desktop/Analytics Software/Group")
#full_data_raw <- read.csv("train_processed.csv")
full_data_raw <- read.csv("0.05subset.csv")
#direct_download_link <- "https://entuedu-my.sharepoint.com/:x:/g/personal/peiyi007_e_ntu_edu_sg/EUwOImJ1cjBArYWP4BCPFFgBPPrBJBi3EFUXrhLbMSFpxw?download=1"
#full_data_raw <- read_csv(direct_download_link)

# 2: data preprocessing
#----------------------------------------------------------------

target_col_name <- "y"
# get the label of first 16 col
cols_to_remove <- names(full_data_raw)[1:16]
# make sure target obj y does not in it
cols_to_remove <- setdiff(cols_to_remove, target_col_name)
# 3. remove col with select function in dplyr
full_data_raw <- full_data_raw %>%
  select(-all_of(cols_to_remove))

cat("remove", length(cols_to_remove), "variables. New dimension:", dim(full_data_raw), "\n\n")


target_col_name <- "y" 
categorical_cols <- c("job", "marital", "education", "housing", 
                      "loan", "contact", "month", "poutcome", "y")

full_data_processed <- full_data_raw %>%
  mutate_at(vars(one_of(categorical_cols)), as.factor)

features_to_encode <- full_data_processed %>% select(-!!sym(target_col_name))
dmy <- dummyVars(" ~ .", data = features_to_encode, fullRank = TRUE)
data_encoded <- data.frame(predict(dmy, newdata = features_to_encode))

final_data <- cbind(data_encoded, y = full_data_processed$y)
cat("preprocessing done\n\n")


# 3 split data frame
#----------------------------------------------------------------
set.seed(42)
train_indices <- createDataPartition(final_data$y, p = 0.8, list = FALSE, times = 1)

train_data <- final_data[train_indices, ]
test_data <- final_data[-train_indices, ]
cat("split done\n\n")


# 4 PCA
#----------------------------------------------------------------
train_x <- train_data %>% select(-!!sym(target_col_name))
train_y <- as.numeric(train_data[[target_col_name]]) - 1
test_x <- test_data %>% select(-!!sym(target_col_name))
test_y <- as.numeric(test_data[[target_col_name]]) - 1

# train PCA model on the training set
pca_model <- prcomp(train_x, center = TRUE, scale. = TRUE)

# Dynamically calculate the number of principal components required to achieve 95% variance
# Calculate the proportion of variance explained by each principal component
variance_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
# Calculate cumulative variance contribution
cumulative_variance <- cumsum(variance_explained)
# Find the minimum number of principal components required to achieve 95% cumulative variance
n_components_95 <- which(cumulative_variance >= 0.95)[1]
# ***************************************************************

cat(paste("To explain 90% of the cumulative variance, it has been automatically selected", n_components_95, "principal components.\n"))

# Use the dynamically calculated n_components_95 to select the principal components
train_x_pca <- as.data.frame(predict(pca_model, newdata = train_x)[, 1:n_components_95])
test_x_pca <- as.data.frame(predict(pca_model, newdata = test_x)[, 1:n_components_95])
cat("PCA done\n\n")


# 5 Efficient automatic hyperparameter tuning
#----------------------------------------------------------------
# Conduct tuning on a subset of the training set to save time
set.seed(123)
sample_size <- 50000 # 50000 sample will be used in the origin set, the whole set can be used if the size is not quite large
if (nrow(train_data) > sample_size) {
  tune_indices <- sample(1:nrow(train_data), size = sample_size)
  train_data_subset_pca <- train_x_pca[tune_indices, ]
  train_y_subset <- train_y[tune_indices]
} else {
  train_data_subset_pca <- train_x_pca
  train_y_subset <- train_y
}
train_data_for_caret <- cbind(train_data_subset_pca, target = factor(train_y_subset, labels = c("no", "yes")))
cat(paste("A subset for tuning has been created, with a size of:", nrow(train_data_for_caret), "\n"))

# Set tuning parameters
# Calculate the sample weights of the subset
spw_subset <- sum(train_y_subset == 0) / sum(train_y_subset == 1)

train_control <- trainControl(
  method = "cv",
  number = 3, # 3-fold cross-validation
  search = "random", # random search
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Run automatic tuning
caret_model <- train(
  target ~ ., 
  data = train_data_for_caret,
  method = "xgbTree",
  trControl = train_control,
  tuneLength = 20, # Try 20 sets of random parameters
  metric = "ROC",
  verbose = FALSE,
  scale_pos_weight = spw_subset
)

cat("automatic tuning done\n")
cat("The optimal parameters found are:\n")
print(caret_model$bestTune)
best_params <- caret_model$bestTune


# 6 final model training
#----------------------------------------------------------------
# Calculate the sample weights for the complete training set
spw_full <- sum(train_y == 0) / sum(train_y == 1)

# DMatrix
dtrain_full <- xgb.DMatrix(data = as.matrix(train_x_pca), label = train_y)
dtest <- xgb.DMatrix(data = as.matrix(test_x_pca), label = test_y)

# set model parameters
final_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best_params$eta,
  max_depth = best_params$max_depth,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  scale_pos_weight = spw_full,
  tree_method = "hist",
  nthread = detectCores() - 1
)

# model training
final_model <- xgb.train(
  params = final_params,
  data = dtrain_full,
  nrounds = best_params$nrounds, # The optimal number of iterations found through tuning
  verbose = 0
)
cat("model training done\n\n")
xgb.save(final_model, "xgboost_model.model")
cat("The model has been successfully saved to the file'xgboost_model.model'。\n\n")

# evaluation and visualization
#----------------------------------------------------------------
# prediction
pred_prob <- predict(final_model, dtest)
pred_class <- factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0, 1))
test_y_factor <- factor(test_y, levels = c(0, 1))

# confusion matrix
cm <- confusionMatrix(pred_class, test_y_factor, positive = "1")
print(cm)
cm_df <- as.data.frame(cm$table)

#Confusion Matrix Heatmap 
heatmap_plot <- ggplot(data = cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") + # geom_tile
  geom_text(aes(label = Freq), vjust = 1, size = 6, color = "white") + 
  scale_fill_gradient(low = "#56B1F7", high = "#132B43") + 
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Predicted Class",
    y = "Actual (Reference) Class",
    fill = "Frequency"
  ) +
  theme_minimal() + 
  theme(
    plot.title = element_text(hjust = 0.5, size=16), 
    axis.title = element_text(size=12),
    axis.text = element_text(size=10)
  )
print(heatmap_plot)

# MCC
TP <- cm$table[2, 2]; TN <- cm$table[1, 1]; FP <- cm$table[2, 1]; FN <- cm$table[1, 2]
mcc_num <- (as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))
mcc_den <- sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
mcc <- ifelse(mcc_den == 0, 0, mcc_num / mcc_den)
cat(paste("\nMCC:", round(mcc, 4), "\n\n"))

# ROC curve and AUC
roc_obj <- roc(test_y, pred_prob, quiet = TRUE)
auc_value <- auc(roc_obj)
cat(paste("AUC:", round(auc_value, 4), "\n"))
plot(roc_obj, main = "ROC Curve", print.auc = TRUE, col = "#1c61b6", lwd = 2)

# PR curve AUC-PR
scores_positive_class <- pred_prob[test_y == 1]
scores_negative_class <- pred_prob[test_y == 0]
pr_curve <- pr.curve(scores.class0 = scores_positive_class, scores.class1 = scores_negative_class, curve = TRUE)
plot(pr_curve, main = "Precision-Recall Curve", auc.main = TRUE)
cat(paste("AUC-PR:", round(pr_curve$auc.integral, 4), "\n\n"))

# Gain Chart and Lift Chart
results_df <- data.frame(prob = pred_prob, actual = test_y) %>% arrange(desc(prob)) %>%
  mutate(decile = ntile(prob, 10))
lift_data <- results_df %>% group_by(decile) %>%
  summarise(n = n(), positives = sum(actual)) %>%
  arrange(decile) %>%
  mutate(cumulative_positives = cumsum(positives), cumulative_n = cumsum(n))
total_positives <- sum(lift_data$positives); total_n <- sum(lift_data$n)
lift_data <- lift_data %>% mutate(gain = cumulative_positives / total_positives, 
                                  cumulative_lift = (cumulative_positives / cumulative_n) / (total_positives / total_n))

# gain Chart
gain_chart_plot <- ggplot(lift_data, aes(x = decile, y = gain)) +
  geom_line(color = "blue", size = 1.2) + geom_point(color = "blue", size = 3) +
  geom_segment(aes(x = 0, y = 0, xend = 10, yend = 1), color = "red", linetype = "dashed") +
  scale_x_continuous(breaks = 0:10) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Gain Chart", x = "Decile", y = "Cumulative % of Positives Captured") + theme_minimal()
print(gain_chart_plot)

# lift chart
lift_chart_plot <- ggplot(lift_data, aes(x = decile, y = cumulative_lift)) +
  geom_bar(stat = "identity", fill = "skyblue") + geom_hline(yintercept = 1, color = "red", linetype = "dashed") +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "Lift Chart", x = "Decile", y = "Cumulative Lift (vs. Random)") + theme_minimal()
print(lift_chart_plot)
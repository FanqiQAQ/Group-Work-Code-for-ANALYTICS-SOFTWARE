# 咳嗽音频疾病预测分析
# 安装所需的包（如果尚未安装）
# install.packages(c("tuneR", "seewave", "randomForest", "caret", "jsonlite", "dplyr", "readr"))

# 加载所需库
library(tuneR)
library(seewave)
library(randomForest)
library(caret)
library(jsonlite)
library(dplyr)
library(readr)
library(reticulate)  # 用于读取.npy文件
library(ggplot2)
library(pROC)

# 设置工作目录和数据路径
setwd("C:/Users/mengfanqi/Desktop/Analytics Software/Group/airs-ai-in-respiratory-sounds") # 根据实际情况修改路径
train_data <- read.csv("train.csv", stringsAsFactors = FALSE)
test_data <- read.csv("test.csv", stringsAsFactors = FALSE)

# 音频特征提取函数
extract_audio_features <- function(audio_file) {
  tryCatch({
    # 读取音频文件
    if (file.exists(audio_file)) {
      wave <- readWave(audio_file)
      
      # 基本音频特征
      features <- list()
      
      # 时域特征
      features$duration <- length(wave@left) / wave@samp.rate
      features$mean_amplitude <- mean(abs(wave@left))
      features$max_amplitude <- max(abs(wave@left))
      features$rms <- sqrt(mean(wave@left^2))
      features$zero_crossing_rate <- sum(diff(sign(wave@left)) != 0) / length(wave@left)
      
      # 频域特征 - 使用seewave包
      # 计算功率谱密度
      spec <- spec(wave, plot = FALSE)
      features$spectral_centroid <- meanspec(wave, plot = FALSE)
      
      # MFCC特征（Mel频率倒谱系数）
      mfcc_result <- melfcc(wave)
      if (!is.null(mfcc_result)) {
        for (i in 1:ncol(mfcc_result)) {
          features[[paste0("mfcc_", i, "_mean")]] <- mean(mfcc_result[, i], na.rm = TRUE)
          features[[paste0("mfcc_", i, "_std")]] <- sd(mfcc_result[, i], na.rm = TRUE)
        }
      }
      
      # 频谱特征
      features$dominant_frequency <- dfreq(wave, plot = FALSE)$y[1]
      features$fundamental_frequency <- fund(wave, plot = FALSE)$y[1]
      
      # 语音质量特征
      features$spectral_rolloff <- specprop(wave, plot = FALSE)$rolloff[1]
      features$spectral_flux <- specprop(wave, plot = FALSE)$flux[1]
      
      return(features)
    } else {
      return(NULL)
    }
  }, error = function(e) {
    cat("处理音频文件时出错:", audio_file, "\n")
    return(NULL)
  })
}

# 读取嵌入特征文件
read_embedding_features <- function(emb_file) {
  tryCatch({
    if (file.exists(emb_file)) {
      emb_data <- fromJSON(emb_file)
      # 如果是列表形式，转换为数值向量
      if (is.list(emb_data)) {
        emb_features <- unlist(emb_data)
      } else {
        emb_features <- as.numeric(emb_data)
      }
      # 创建命名列表
      feature_names <- paste0("emb_", seq_along(emb_features))
      features <- as.list(emb_features)
      names(features) <- feature_names
      return(features)
    } else {
      return(NULL)
    }
  }, error = function(e) {
    cat("读取嵌入文件时出错:", emb_file, "\n")
    return(NULL)
  })
}

# 读取.npy文件的函数（需要安装reticulate包）
read_npy_features <- function(npy_file) {
  tryCatch({
    if (file.exists(npy_file)) {
      # 使用reticulate调用numpy读取.npy文件
      np <- import("numpy")
      npy_data <- np$load(npy_file)
      
      # 如果是多维数组，计算统计特征
      if (length(dim(npy_data)) > 1) {
        features <- list(
          opera_mean = mean(npy_data, na.rm = TRUE),
          opera_std = sd(as.vector(npy_data), na.rm = TRUE),
          opera_max = max(npy_data, na.rm = TRUE),
          opera_min = min(npy_data, na.rm = TRUE),
          opera_median = median(npy_data, na.rm = TRUE)
        )
      } else {
        # 一维数组，直接使用
        features <- list()
        for (i in seq_along(npy_data)) {
          features[[paste0("opera_", i)]] <- npy_data[i]
        }
      }
      return(features)
    } else {
      return(NULL)
    }
  }, error = function(e) {
    cat("读取.npy文件时出错:", npy_file, "\n")
    return(NULL)
  })
}

# 为每个候选人提取特征的函数
extract_candidate_features <- function(candidate_id, base_path = ".") {
  folder_path <- file.path(base_path, as.character(candidate_id))
  
  all_features <- list()
  
  # 1. 咳嗽音频特征
  cough_wav <- file.path(folder_path, "cough.wav")
  cough_features <- extract_audio_features(cough_wav)
  if (!is.null(cough_features)) {
    names(cough_features) <- paste0("cough_", names(cough_features))
    all_features <- c(all_features, cough_features)
  }
  
  # 2. 元音音频特征
  vowel_wav <- file.path(folder_path, "vowel.wav")
  vowel_features <- extract_audio_features(vowel_wav)
  if (!is.null(vowel_features)) {
    names(vowel_features) <- paste0("vowel_", names(vowel_features))
    all_features <- c(all_features, vowel_features)
  }
  
  # 3. 咳嗽嵌入特征
  cough_emb <- file.path(folder_path, "emb_cough.json")
  cough_emb_features <- read_embedding_features(cough_emb)
  if (!is.null(cough_emb_features)) {
    names(cough_emb_features) <- paste0("cough_", names(cough_emb_features))
    all_features <- c(all_features, cough_emb_features)
  }
  
  # 4. 元音嵌入特征
  vowel_emb <- file.path(folder_path, "emb_vowel.json")
  vowel_emb_features <- read_embedding_features(vowel_emb)
  if (!is.null(vowel_emb_features)) {
    names(vowel_emb_features) <- paste0("vowel_", names(vowel_emb_features))
    all_features <- c(all_features, vowel_emb_features)
  }
  
  # 5. Opera特征文件
  cough_opera <- file.path(folder_path, "cough-opera.npy")
  cough_opera_features <- read_npy_features(cough_opera)
  if (!is.null(cough_opera_features)) {
    names(cough_opera_features) <- paste0("cough_", names(cough_opera_features))
    all_features <- c(all_features, cough_opera_features)
  }
  
  vowel_opera <- file.path(folder_path, "vowel-opera.npy")
  vowel_opera_features <- read_npy_features(vowel_opera)
  if (!is.null(vowel_opera_features)) {
    names(vowel_opera_features) <- paste0("vowel_", names(vowel_opera_features))
    all_features <- c(all_features, vowel_opera_features)
  }
  
  return(all_features)
}

# 处理训练数据
cat("开始处理训练数据...\n")
train_features <- data.frame()

for (i in 1:nrow(train_data)) {
  candidate_id <- train_data$candidateID[i]
  cat("处理训练候选人:", candidate_id, "(", i, "/", nrow(train_data), ")\n")
  
  # 提取音频特征
  audio_features <- extract_candidate_features(candidate_id)
  
  # 合并所有特征
  row_features <- train_data[i, ]
  
  # 添加音频特征
  if (length(audio_features) > 0) {
    audio_df <- data.frame(t(unlist(audio_features)), stringsAsFactors = FALSE)
    row_features <- cbind(row_features, audio_df)
  }
  
  # 添加到训练特征数据框
  if (nrow(train_features) == 0) {
    train_features <- row_features
  } else {
    # 确保列名一致
    common_cols <- intersect(names(train_features), names(row_features))
    missing_in_train <- setdiff(names(row_features), names(train_features))
    missing_in_row <- setdiff(names(train_features), names(row_features))
    
    # 添加缺失的列
    if (length(missing_in_train) > 0) {
      train_features[missing_in_train] <- NA
    }
    if (length(missing_in_row) > 0) {
      row_features[missing_in_row] <- NA
    }
    
    train_features <- rbind(train_features, row_features)
  }
}

# 处理测试数据
cat("开始处理测试数据...\n")
test_features <- data.frame()

for (i in 1:nrow(test_data)) {
  candidate_id <- test_data$candidateID[i]
  cat("处理测试候选人:", candidate_id, "(", i, "/", nrow(test_data), ")\n")
  
  # 提取音频特征
  audio_features <- extract_candidate_features(candidate_id)
  
  # 合并所有特征
  row_features <- test_data[i, ]
  
  # 添加音频特征
  if (length(audio_features) > 0) {
    audio_df <- data.frame(t(unlist(audio_features)), stringsAsFactors = FALSE)
    row_features <- cbind(row_features, audio_df)
  }
  
  # 添加到测试特征数据框
  if (nrow(test_features) == 0) {
    test_features <- row_features
  } else {
    # 确保列名一致
    common_cols <- intersect(names(test_features), names(row_features))
    missing_in_test <- setdiff(names(row_features), names(test_features))
    missing_in_row <- setdiff(names(test_features), names(row_features))
    
    # 添加缺失的列
    if (length(missing_in_test) > 0) {
      test_features[missing_in_test] <- NA
    }
    if (length(missing_in_row) > 0) {
      row_features[missing_in_row] <- NA
    }
    
    test_features <- rbind(test_features, row_features)
  }
}

# 数据预处理
cat("开始数据预处理...\n")

# 确保训练和测试数据有相同的列（除了disease列）
train_cols <- names(train_features)
test_cols <- names(test_features)

# 添加disease列到测试数据（用于保持一致性）
if (!"disease" %in% test_cols) {
  test_features$disease <- NA
}

# 确保列的一致性
all_cols <- unique(c(train_cols, test_cols))
for (col in all_cols) {
  if (!col %in% train_cols) {
    train_features[[col]] <- NA
  }
  if (!col %in% test_cols) {
    test_features[[col]] <- NA
  }
}

# 重新排列列的顺序
train_features <- train_features[, sort(names(train_features))]
test_features <- test_features[, sort(names(test_features))]

# 处理缺失值和无穷值
numeric_cols <- sapply(train_features, is.numeric)
train_features[numeric_cols] <- lapply(train_features[numeric_cols], function(x) {
  x[is.infinite(x)] <- NA
  x[is.nan(x)] <- NA
  return(x)
})

test_features[numeric_cols] <- lapply(test_features[numeric_cols], function(x) {
  x[is.infinite(x)] <- NA
  x[is.nan(x)] <- NA
  return(x)
})

# 用中位数填充数值型变量的缺失值
for (col in names(train_features)[numeric_cols]) {
  if (col != "disease") {
    median_val <- median(train_features[[col]], na.rm = TRUE)
    if (!is.na(median_val)) {
      train_features[[col]][is.na(train_features[[col]])] <- median_val
      test_features[[col]][is.na(test_features[[col]])] <- median_val
    }
  }
}

# 导出特征文件
cat("导出特征文件...\n")
write.csv(train_features, "train_features.csv", row.names = FALSE)
write.csv(test_features, "test_features.csv", row.names = FALSE)

# 合并所有特征并导出
all_features <- rbind(train_features, test_features)
write.csv(all_features, "features.csv", row.names = FALSE)

# 准备机器学习模型
cat("开始训练模型...\n")

# 选择特征列（排除ID和目标变量）
feature_cols <- setdiff(names(train_features), c("candidateID", "disease"))
X_train <- train_features[, feature_cols]
y_train <- as.factor(train_features$disease)
X_test <- test_features[, feature_cols]

# 移除常数列和方差为0的列
var_check <- sapply(X_train, function(x) var(x, na.rm = TRUE))
constant_cols <- names(var_check)[var_check == 0 | is.na(var_check)]
if (length(constant_cols) > 0) {
  X_train <- X_train[, !names(X_train) %in% constant_cols]
  X_test <- X_test[, !names(X_test) %in% constant_cols]
}

# 训练随机森林模型
set.seed(123)
rf_model <- randomForest(X_train, y_train, 
                         ntree = 500, 
                         importance = TRUE,
                         na.action = na.omit)

# 预测测试数据
predictions <- predict(rf_model, X_test)

# 创建提交文件
submission <- data.frame(
  candidateID = test_features$candidateID,
  predicted_disease = predictions
)

write.csv(submission, "predictions.csv", row.names = FALSE)

# 模型评估和特征重要性
cat("模型训练完成!\n")
cat("特征重要性排序:\n")
importance_scores <- importance(rf_model)
sorted_importance <- importance_scores[order(importance_scores[,1], decreasing = TRUE), ]
print(head(sorted_importance, 20))


# 保存模型
save(rf_model, file = "cough_disease_model.RData")

cat("分析完成! 文件已保存:\n")
cat("- features.csv: 所有提取的特征\n")
cat("- train_features.csv: 训练数据特征\n")
cat("- test_features.csv: 测试数据特征\n")
cat("- predictions.csv: 测试数据预测结果\n")
cat("- cough_disease_model.RData: 训练好的模型\n")
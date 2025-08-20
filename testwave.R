library(tuneR)
library(seewave)
library(soundgen)
# 读取音频文件
setwd("C:/Users/mengfanqi/Desktop/Analytics Software/Group/00ad476fba37b")
wave <- readWave("cough.wav")

# 转换为单声道（简化处理）
mono_wave <- mono(wave, "left")

# 信号和采样率
signal <- mono_wave@left
fs <- mono_wave@samp.rate  # 采样率(Hz)

#FFT
fft_result <- fft(signal)

# 计算幅度谱
magnitude <- Mod(fft_result)

# 创建频率轴
n <- length(signal)
freq <- seq(0, fs/2, length.out = n/2 + 1)

# 取一半（对称性）
magnitude_onesided <- magnitude[1:(n/2 + 1)]

# 绘制频谱图
plot(freq, magnitude_onesided, type = "l", 
     xlab = "Frequency (Hz)", ylab = "Magnitude",
     main = "Frequency Spectrum")

#频谱分析
# 使用seewave绘制频谱图
spectro(mono_wave, f = fs, 
        flim = c(0, 5),   # 频率范围0-5kHz
        palette = reverse.terrain.colors)

# 设计低通滤波器（保留<1kHz）
filtered <- bwfilter(mono_wave, f = fs, 
                     n = 4,        # 滤波器阶数
                     to = 1000,    # 截止频率
                     bandpass = FALSE)

# 比较原始和滤波后频谱
par(mfrow = c(2, 1))
spectro(mono_wave, f = fs, flim = c(0, 5), main = "Original")
spectro(filtered, f = fs, flim = c(0, 5), main = "Filtered")

# 使用自相关函数检测基频
pitch <- autoc(mono_wave, f = fs)
cat("Estimated pitch:", pitch$pitch, "Hz\n")

# 梅尔频率倒谱系数（语音识别关键特征）
mfccs <- melfcc(mono_wave, sr = fs, 
                #n_mfcc = 13,    # 系数数量
                wintime = 0.025, # 窗口长度25ms
                hoptime = 0.01)  # 步长10ms

# 可视化MFCC
image(t(mfccs), 
      xlab = "Time Frames", 
      ylab = "MFCC Coefficients",
      main = "MFCC Features")
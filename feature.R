library(tuneR)
library(seewave)
library(soundgen)

extract_audio_features <- function(file_path) {
  # load wav
  wave <- readWave(file_path)
  mono_wave <- mono(wave, "left")
  fs <- mono_wave@samp.rate
  signal <- mono_wave@left
  
  # basic feature
  duration <- length(signal)/fs
  zcr <- zcr(signal, fs)  # Zero Crossing Rate
  rms <- rms(signal)      # Root Mean SquareEnergy
  
  # frequency feature
  spec <- meanspec(mono_wave, f = fs, plot = FALSE)
  spectral_centroid <- mean(spec[,1] * spec[,2]) / mean(spec[,2])
  
  # MFCC
  mfccs <- melfcc(mono_wave, sr = fs)
  mfcc_mean <- colMeans(mfccs)
  
  # combine features
  features <- c(
    duration = duration,
    zcr = zcr,
    rms = rms,
    spectral_centroid = spectral_centroid,
    mfcc_mean = mfcc_mean
  )
  
  return(features)
}

# example
features<-extract_audio_features("C:/Users/mengfanqi/Desktop/Analytics Software/Group/00ad476fba37b/cough.wav")
head(features)
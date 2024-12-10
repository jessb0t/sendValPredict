# sendValPredict: LLM analyses
# Author: Jessica M. Alexander
# Last Updated: 2024-12-08
# Input:
#     --preprocessed, aggregated data (llm_prediction_data.csv)
# Output:
#     For LLM predictions: plot of human valence ratings for each 5-second window in the dataset, including:
#       -coefficient of determination (r-squared)
#       -pearson correlation coefficient
#       -concordance correlation coefficient


# SETUP
##load libraries
library(tidyverse)
library(gridExtra)
library(viridis)

#helper functions
rsquared <- function(x, y){
  ssr <- sum((x-y)^2)
  sst <- sum((x-mean(x))^2)
  r2 <- 1 - (ssr / sst)
  return (r2)
}

pcc <- function(x, y){
  pearson_numerator <- sum((x - mean(x)) * (y - mean(y)))
  pearson_denominator <- sqrt(sum((x - mean(x))**2) * sum((y - mean(y))**2))
  pearson_p <- pearson_numerator / pearson_denominator
  return (pearson_p)
}

ccc <- function(x, y){
  ccc_numerator <- 2 * pcc(x, y) * sd(x) * sd(y)
  ccc_denominator <- (sd(x)**2) + (sd(y)**2) + (mean(x) - mean(y))**2
  ccc <- ccc_numerator / ccc_denominator
  return (ccc)
}

digit_display <- function(number){
  if(abs(number)<0.00001){
    x <- sprintf("%.6f", number)
  }else{
    x <- sprintf("%.3f", number)
  }
  return (x)
}

##load all data
version = "pre-tune" #c("pre-tune", "post-tune")
if(version == "pre-tune"){
  df <- read.csv("llm_prediction_data.csv") #out-of-sample predictions before fine-tuning (all windows)
} else if(version == "post-tune"){
  df <- read.csv("llm_finetuned_prediction_data.csv") #out-of-sample predictions after fine-tuning (subset of windows not used for fine-tuning)
}

df$ewe <- df$ewe*100
df$valpred <- df$valpred*100

#calculate metrics: pearson correlation coefficient (pearson_p), concordance correlation coefficient (ccc), R2
#evaluation metrics
r2 <- rsquared(df$ewe, df$valpred)
pearson_p <- pcc(df$ewe, df$valpred)
ccc <- ccc(df$ewe, df$valpred)

#visualize
label1 <- paste("R^2 == ", digit_display(r2), sep="")
label2 <- paste("Pearson's r = ", digit_display(pearson_p), sep="")
label3 <- paste("CCC = ", digit_display(ccc), sep="")

plot <- df %>%
  ggplot(aes(x=ewe, y=valpred)) +
  geom_point(data=df, aes(color=spkr), size=0.5) +
  scale_color_viridis(discrete=TRUE, option="viridis") +
  labs(title="MSP-Podcast Fine-Tuned Wav2Vec2.0 v. Human Raters",
       x="Human Valence Rating", y="LLM Valence Rating") +
  theme_classic() +
  theme(legend.position = "none") +
  ylim(c(0,100)) + 
  annotate("text", x=6, y=95, label=label1, size=3, hjust=0, parse=TRUE) +
  annotate("text", x=6, y=90, label=label2, size=3, hjust=0) +
  annotate("text", x=6, y=85, label=label3, size=3, hjust=0)

if(version == "post-tune"){
  plot <- plot + labs(subtitle="after fine-tuning on 50% of SEND corpus speakers")
}

#ggsave("figs/llm_results.png", height=4, width=6, units="in")


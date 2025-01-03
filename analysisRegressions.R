# sendValPredict: linear regression analyses
# Author: Jessica M. Alexander
# Last Updated: 2025-01-03
# Input:
#     --preprocessed, aggregated data (data.csv)
# Output:
#     For four models:
#         *warriner: averaged lexical valence and lexical arousal (from Warriner et al., 2013)
#         *glove: 300-dimension centroid of GloVe embeddings
#         *acoustics: 88 dimensions of extracted eGeMAPS acoustic features (from Eyben et al., 2016),
#                     mapped onto 88 principal components to remove collinearity
#         *combo: combination of all predictors for both glove and acoustics models
#     --plot of human valence ratings for each 5-second window in the dataset, including:
#       -coefficient of determination (r-squared)
#       -pearson correlation coefficient
#       -concordance correlation coefficient
#     In line, beta weights for statistically significant predictors for
#     glove, acoustics, and combo models can be explored
# Usage Notes:
#     --set iterations parameter, then run script

# SETUP
set.seed(2024)

##select number of iterations
iterations = 1000

##load libraries
library(tidyverse)
library(lme4)
library(gridExtra)
library(viridis)

##helper functions
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
df <- read.csv("data.csv")

##drop rows rows with missing/infinite values (when unable to calculate either acoustic feature or glove centroid)
rows_to_drop <- c()
for(x in 1:nrow(df)){
  val <- sum(!is.finite(as.numeric(as.vector(df[x,c(6:ncol(df))]))))
  if(val>0){
    rows_to_drop <- c(rows_to_drop, x)
  }
}
df <- df[-rows_to_drop,]

##drop first five seconds of each narrative (likely that ratings in first 5 seconds represent a lag in rater reactions)
firstfivesecs <- which(df$time==0)
df <- df[-firstfivesecs,]

##convert speaker to factor
df$spkr <- as.factor(df$spkr)

##visually check distribution of human ratings
plot1 <- df %>%
  ggplot(aes(x=ewe, y=0, color=spkr)) + 
  geom_jitter(height=0.25, size=0.25) +
  ylim(c(-0.3, 0.3)) +
  scale_color_viridis(discrete=TRUE, option="viridis") +
  labs(title="Distribution of Human Valence Ratings",
       subtitle="for each 5-second window",
       x="", y="") +
  theme_void() +
  theme(legend.position = "none")

plot2 <- df %>%
  ggplot(aes(x=ewe)) + geom_density() +
  xlim(c(0,100)) +
  labs(x="Valence Rating (per 5-second window)", y="") +
  theme_classic() +
  theme(axis.line.y = element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

comboPlot <- gridExtra::grid.arrange(plot1, plot2,
                                     layout_matrix = matrix(c(1,1,2), byrow = TRUE, ncol = 1))
#output combo: RStudio Export>Save as Image (693 x 352) >> "dist-val-ratings.png"

# ORGANIZE BOOTSTRAPPING DATAFRAMES
#warriner
warrinercolnames <- colnames(df)[c(6:7)]
a <- tibble(i=rep(1:iterations,))
a <- mutate(a, intercept=NA)
a <- cbind(a, setNames( lapply(warrinercolnames, function(x) x=NA), warrinercolnames))

#glove
glovecolnames <- colnames(df)[c(97:ncol(df))]
b <- tibble(i=rep(1:iterations,))
b <- mutate(b, intercept=NA)
b <- cbind(b, setNames( lapply(glovecolnames, function(x) x=NA), glovecolnames))

#acoustics
#with thanks to Gavin Simpson for this post: https://stackoverflow.com/questions/12760108/principal-components-analysis-how-to-get-the-contribution-of-each-paramete
acousticDat <- df[,c(9:96)]
acoustic.pca <- stats::prcomp(acousticDat, scale=TRUE)
#summary(acoustic.pca)

acoustic.pca.x <- as.data.frame(acoustic.pca$x)
acousticolnames <- colnames(acoustic.pca.x)
df <- cbind(df, acoustic.pca.x)
c <- tibble(i=rep(1:iterations,))
c <- mutate(c, intercept=NA)
c <- cbind(c, setNames( lapply(acousticolnames, function(x) x=NA), acousticolnames))

#combo glove+acoustics
combocolnames <- c(glovecolnames, acousticolnames)
d <- tibble(i=rep(1:iterations,))
d <- mutate(d, intercept=NA)
d <- cbind(d, setNames( lapply(combocolnames, function(x) x=NA), combocolnames))


# BOOTSTRAP IT!
for (i in 1:iterations){
  sampled_rows <- sample(nrow(df), nrow(df), replace=TRUE)
  data <- df[sampled_rows,]
  
  #warriner ratings
  print(paste("running warriner model, iteration ", as.character(i)))
  mdl <- lmer(ewe ~ valMean + aroMean + (1|spkr), data=data)
  a[i,2:ncol(a)] <- as.list(fixef(mdl))

  #glove embeddings
  print(paste("running glove model, iteration ", as.character(i)))
  glovecols <- data[,c(2, 8, 97:396)]
  mdl <- lmer(ewe ~ . -spkr + (1|spkr), data=glovecols)
  b[i,2:ncol(b)] <- as.list(fixef(mdl))
  
  #acoustic features
  print(paste("running acoustic model, iteration ", as.character(i)))
  acousticcols <- data[,c(2, 8, 397:484)]
  mdl <- lmer(ewe ~ . -spkr + (1|spkr), data=acousticcols)
  c[i,2:ncol(c)] <- as.list(fixef(mdl))
  
  #combo glove + acoustic
  print(paste("running combo model, iteration ", as.character(i)))
  combocols <- data[,c(2, 8, 97:484)]
  mdl <- lmer(ewe ~ . -spkr + (1|spkr), data=combocols)
  d[i,2:ncol(d)] <- as.list(fixef(mdl))
}

# FUNCTIONS FOR MODEL COMPARISON
##function to calculate evaluation metrics and build plot
evaluate_and_plot <- function(dataframe, databoot, title){
  betas <- colMeans(databoot)
  intercept <- betas[2]
  predictors <- colnames(databoot)[3:ncol(databoot)]
  predbetas <- betas[3:length(betas)]
  
  relevantcols <- names(dataframe)[which(names(dataframe) %in% names(databoot))]
  trueDat <- select(dataframe, all_of(c("spkr", "ewe", relevantcols)))
  
  trueDat$pred <- c()
  for(i in 1:nrow(trueDat)){
    y <- intercept
    for(j in 1:length(relevantcols)){
      thisbeta <- predbetas[j]
      val <- trueDat[i, relevantcols[j]]
      addl <- thisbeta * val
      y <- y + addl
    }
    trueDat$pred[i] <- y
  }
  
  #evaluation metrics
  r2 <- rsquared(trueDat$ewe, trueDat$pred)
  pearson_p <- pcc(trueDat$ewe, trueDat$pred)
  ccc <- ccc(trueDat$ewe, trueDat$pred)
  
  label1 <- paste("R^2 == ", digit_display(r2), sep="")
  label2 <- paste("Pearson's r = ", digit_display(pearson_p), sep="")
  label3 <- paste("CCC = ", digit_display(ccc), sep="")
  
  plot <- ggplot(data=trueDat, aes(x=ewe, y=pred)) +
    geom_point(data=trueDat, aes(color=spkr), size=0.5) +
    scale_color_viridis(discrete=TRUE, option="viridis") +
    labs(title=title,
         x="Human Valence Rating", y="Predicted Valence Rating") +
    theme_classic() +
    theme(legend.position = "none") +
    ylim(c(0,100)) + 
    annotate("text", x=6, y=95, label=label1, size=3, hjust=0, parse=TRUE) +
    annotate("text", x=6, y=90, label=label2, size=3, hjust=0) +
    annotate("text", x=6, y=85, label=label3, size=3, hjust=0)
  
  return (plot)
}

##function to calculate significance for each beta
calculate_significance <- function(databoot){
  betas <- colMeans(databoot)
  stderr <- sapply(databoot, function(x) var(x))
  tval <- betas / stderr
  tval_abs <- abs(tval)
  pvals <- lapply(tval_abs, function(q) 2 * pt(q, nrow(databoot), lower.tail=FALSE))
  allps <- unlist(pvals)[3:length(pvals)] #drop "i" column and intercept
  betas <- betas[3:length(betas)] #drop intercept
  return (betas[allps < 0.05])
}

##############################################################
# COMPARE MODELS
evaluate_and_plot(df, a, "Lexical Valence+Arousal v. Human Raters")
ggsave("figs/warriner_results.png", height=4, width=6, units="in")

evaluate_and_plot(df, b, "GloVe Embeddings v. Human Raters")
ggsave("figs/glove_results.png", height=4, width=6, units="in")

evaluate_and_plot(df, c, "Acoustic Feature PCAs v. Human Raters")
ggsave("figs/acoustics_results.png", height=4, width=6, units="in")

evaluate_and_plot(df, d, "Combo GloVe + Acoustic PCAs v. Human Raters")
ggsave("figs/combo_results.png", height=4, width=6, units="in")


# EXPLORE BETA WEIGHTS
#GloVe model: beta weights for dimensions with p<0.05:
betas_glove <- calculate_significance(b)
round(betas_glove, 3)

#Acoustics model: beta weights for principal components with p<0.05:
betas_acoustic <- calculate_significance(c)
round(betas_acoustic, 3)

#Combo model: beta weights for GloVe dimensions and acoustic PCs with p<0.05:
betas_combo <- calculate_significance(d)
round(betas_combo, 3)
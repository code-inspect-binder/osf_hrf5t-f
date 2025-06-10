# Run the GCM JAGS model

# ---- Library calls ----
library(openxlsx)
library(pdist)
library(rjags)
library(runjags)
library(plyr)

# ---- Load and preprocess the data ----
# This section loads in my full dataset from my dissertation into the variable 'data'
# I subset the dataset to only the test blocks, then create sequential subject IDs and 
# rock numbers. The result is a dataset with 44274 rows of 14 different variables.
load("fullDataClean.rds")
data = numOnlyClean; rm(numOnlyClean)
data = subset(data, block == 4)
data = data[order(data$subid),]
data$seqid = cumsum(!duplicated(data$subid))
data$stimid = as.numeric(factor(data$rocknumber))

# Read in coordinates of each rock
coordinates = read.xlsx("540_coordinates.xlsx", colNames = TRUE)
coordinates = as.matrix(unname(coordinates))
coordinates = coordinates[,2:9]

# List indices of training stimuli
train = c(1,	11,	14,	12,	9,	16, 
          26,	27,	31,	28,	22,	19, 
          44,	39,	35,	33,	34,	45, 
          60,	57,	50,	51,	55,	58, 
          70,	71,	80,	67,	73,	76, 
          82,	96,	81,	85,	88,	86, 
          101, 106, 105, 102, 99, 103, 
          125, 126, 116, 119,	118, 121, 
          138, 129, 139, 131,	130, 135, 
          152, 151, 146, 155,	147, 149, 
          172, 169, 162, 174,	168, 166, 
          177, 188, 190, 189,	184, 187, 
          197, 201, 202, 206,	207, 208, 
          219, 213, 210, 221,	212, 224, 
          229, 225, 232, 231,	234, 227, 
          251, 247, 255, 249,	242, 256, 
          269, 266, 261, 263,	268, 264, 
          274, 279, 285, 284,	283, 278, 
          304, 299, 297, 301,	296, 289, 
          313, 315, 319, 305,	310, 318, 
          321, 336, 326, 322,	331, 329, 
          339, 349, 344, 338,	352, 351, 
          368, 359, 367, 365,	362, 356, 
          371, 379, 380, 383,	378, 376, 
          393, 398, 385, 391,	396, 389, 
          407, 410, 404, 408,	413, 402, 
          426, 425, 429, 427,	419, 422, 
          441, 446, 444, 440,	447, 437, 
          451, 450, 463, 454,	459, 452, 
          469, 476, 473, 466,	465, 474)
train = as.integer(train)
ncat = 3

# Determine prototypes of each category
prototypes = matrix(c(colMeans(coordinates[train[1:60],]),
                      colMeans(coordinates[train[61:120],]),
                      colMeans(coordinates[train[121:180],])),
                    nrow = 3, ncol = 8, byrow = TRUE)

# Creates the variable 'distances', a 540 X 3 distance matrix between stimuli and 
# category prototypes.
distances = as.matrix(pdist(coordinates, prototypes))

# ---- Set up and sample the model ----
#######################################################
# Variables supplied in 'data' argument to run.jags: 
# y: Confusion matrix of proportions, nstim X ncat
# d: Matrix of distances between training stimuli, as calculated from our MDS solution
# ncat: Number of categories in the experiment
# nsubj: Number of subjects in the experiment
# nrows: Number of individual judgments (i.e. rows in the dataset)
# subid: Sequential subject id numbers
# stimid: Sequential stimulus numbers, for indexing purposes
#######################################################

model = run.jags(model = "HierarchicalPrototype.R",
                 data = list(
                   'y' = as.numeric(data$catresponse),
                   'd' = distances,
                   'ncat' = ncat,
                   'nsubj' = length(unique(data$subid)),
                   'nrows' = nrow(data),
                   'subid' = data$seqid,
                   'stimid' = data$stimid
                 ),
                 monitor = c('chyper','c','predy'),
                 n.chains = 4,
                 adapt = 500,
                 burnin = 0,
                 sample = 1000,
                 keep.jags.files = TRUE)

# ---- View results ----
# Summarize values of all monitored variables and store into the variable "values"
values = summary(model, vars = "c")
# Create plots of the c parameters
plot(model, vars = "c")

# ---- Read model and process predictions ----
# Collapse by rock number and plot predicted vs. observed

# Lines 111-134 dig into the MCMC samples and give back 'propy',
# a dataframe containing the predicted proportions with which each 
# stimulus will be categorized into each category across all subjects
npreds = nrow(data)
predy = matrix(nrow = npreds, ncol = ncat)
nrocks = length(unique(data$rocknumber))

preds = character(npreds)
for (i in 1:npreds){
  preds[i] = paste0("predy[", i, "]")
}

chainvals = model$mcmc[,preds]
for (i in 1:npreds){
  for (j in 1:ncat){
    predy[i,j] = sum(unlist(chainvals[,preds[i]]) == j)
  }
}
predy = as.data.frame(predy); colnames(predy) = c("igneous", "metamorphic", "sedimentary")
predy$rocknumber = data$rocknumber
predy$category = data$category

propy = ddply(predy, .(rocknumber), .fun = function(x){
  data.frame(pIgneous = sum(x[,1])/sum(x[,1:3]),
             pMetamorphic = sum(x[,2])/sum(x[,1:3]),
             pSedimentary = sum(x[,3])/sum(x[,1:3]))
})

# Lines 137-151 read in the actual aggregated rock data
aggData = read.xlsx("aggregateRockData.xlsx", colNames = FALSE)
colnames(aggData) = c("rocknum",
                   "category",
                   "subtype",
                   "token",
                   "itemType",
                   "freqIgneous",
                   "freqMetamorphic",
                   "freqSedimentary",
                   "freqOld",
                   "freqNew",
                   "pIgneous",
                   "pMetamorphic",
                   "pSedimentary",
                   "pOld")

# Create a predicted vs. observed plot
numObs = rowSums(cbind(aggData$freqIgneous,aggData$freqMetamorphic,aggData$freqSedimentary))
pred = as.matrix(propy[,2:4] * numObs)
obs = as.matrix(cbind(aggData$freqIgneous,aggData$freqMetamorphic,aggData$freqSedimentary))
plot(pred, obs); abline(0,1); title("Hierarchical Prototype")
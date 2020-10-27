# Title: Discover Associations Between Products

# Last update: 3-2-2020

# File/project name: Task4_ProductAssociations
# RStudio Project name: Task4_ProductAssociations

###############
# Project Notes
###############


# Summarize project: This is a simple R pipeline to help understand how to organize
# a project in R Studio using the the caret package. This is a basic pipeline that
# does not include feature selection (filtering/wrapper methods/embedded methods).  



# Objects


# Assignment "<-" short-cut: 
#   OSX [Alt]+[-] (next to "+" sign)
#   Win [Alt]+[-] 


# Comment multiple lines
# OSX: CTRL + SHIFT + C
# WIN: CMD + SHIFT + C


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
# set working directory
setwd("D:/UT Data Analytics/Course 2 - Predicting Customer Preferences/Task4 - Discover Associations Between Products/Task4_ElectrondexProductAssociations")
dir()

# set a value for seed (to be used in the set.seed function)
seed <- 123


################
# Load packages
################
install.packages("doParallel") # install in 'Load packages' section above
install.packages("arules")
install.packages("arulesViz")
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("plotly")
library(arules)
library(arulesViz)
library(caret)
library(corrplot)
#library(doMC)
library(doParallel)
library(mlbench)
library(readr)


#####################
# Parallel processing
#####################

# NOTE: Be sure to use the correct package for your operating system.


#--- for WIN ---#

detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores; 2 in this example
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)

dev.off()

##############
# Import data
##############

#### --- Load all raw datasets --- ####

# --- Load Train/Existing data (Dataset 1) --- #

transactionData <- read.transactions("ElectronidexTransactions2017.csv", format = "basket", sep = ",", rm.duplicates = TRUE)

#### --- Load all preprocessed datasets --- ####

################
# Evaluate data
################

#--- Dataset 1 ---#

class(transactionData)  # "data.frame"
str(transactionData)
inspect(transactionData)# You can view the transactions. Is there a way to see a certain # of transactions?
length(transactionData)# Number of transactions.
transactionSize <- size(transactionData)# Number of items per transaction
LIST(transactionData)# Lists the transactions by conversion (LIST must be capitalized)
itemLabels(transactionData)# To see the item labels
names(transactionData)
summary(transactionData)
head(transactionData)
tail(transactionData)


# plot
#hist()
plot(transactionSize, main = "Transaction Sizes", type = "p", cex = .4, col="dark blue")
plot(sample(transactionSize, 500), main = "Sample Transaction Sizes", cex = .4, col="dark red")

frequentItems <- eclat(transactionData, parameter = list(supp=0.1, maxlen = 15))
inspect(sort(frequentItems))

itemFrequencyPlot(transactionData, topN=15, type = "absolute", horiz = TRUE, main = "Item Frequency")
itemFrequencyPlot(transactionData, topN=10, type = "absolute", main = "Item Frequency")
itemFrequencyPlot(transactionData, topN=20, type = "absolute", main = "Item Frequency")
itemFrequencyPlot(head(transactionData), type = "absolute",main = "Item Frequency")
itemFrequencyPlot(transactionData, topN=50, type = "absolute", main = "Item Frequency")

image(sample(transactionData, 1000))



### Build Association Rules ###

#Out of Box Dataset Rules#


rules <- apriori (transactionData, parameter = list(supp = 0.001,minlen = 2, conf = 0.8))
rules.sorted <- sort(rules, by="lift")
rulesRedun <- is.redundant(rules.sorted)

#Find Redundant Rules#
subset.matrix <- is.subset(rules.sorted, rules.sorted, sparse=FALSE)
subset.matrix[lower.tri(subset.matrix, diag = T)] <- NA
redundant <- colSums(subset.matrix, na.rm = T) >= 1
which(redundant)
#Remove Redundant Rules#
rules.pruned <- rules.sorted[!redundant]
is.redundant(rules.pruned)

###New rules without redundancy##
summary(rules.pruned)
inspect(rules.pruned)
inspect(sort(rules.pruned, by = "confidence"))
inspect(sample(sort(rules.pruned, by = "confidence"),5))
inspect(head(sort(rules.pruned, by="confidence")))

plot(rules.pruned, measure = c("confidence","lift"))
plot(rules.pruned, measure = c("support","lift"))
plot(rules.pruned, measure = c("support","confidence"))

plot(rules.pruned[1:10], method="graph", control=list(type="b")) 
head(quality(rules.pruned))
plot(rules.pruned, measure = c("support","lift"),shading = "confidence")
plot(rules.pruned, method = "two-key plot")
plot(rules.pruned,method = "grouped", main="")
inspectDT(rules.pruned)
ruleExplorer(rules.pruned)
plot(rules.pruned, engine = "plotly")


subConf <- rules.pruned[(quality(rules.pruned)$confidence) > 0.6]
subConf
is.redundant(subConf)
plot(subConf, method = "matrix", measure = "lift")
plot(subConf, method = "matrix3D", measure = "lift")



### Confidence 
aprioriRulesConf <- sort(rules.pruned, by = "confidence", decreasing = TRUE)
inspect(aprioriRulesConf)
plot(aprioriRulesConf)
plot(aprioriRulesConf, measure = c("support","lift"),shading = "confidence")
plot(aprioriRulesConf, method = "two-key plot")
inspectDT(aprioriRulesConf)
is.redundant(aprioriRulesConf)

aprioriRulesLift <- sort(rules.pruned, by = "lift", decreasing = TRUE)
inspect(aprioriRulesLift)
plot(aprioriRulesLift)
plot(aprioriRulesLift, measure = c("support","lift"),shading = "confidence")
plot(aprioriRulesLift, method = "two-key plot")
inspectDT(aprioriRulesLift)
is.redundant(aprioriRulesLift)

##Desktop Rules##
desktopRules <- subset(rules.pruned, subset = rhs %pin% "Desktop")
inspect(desktopRules)
plot(desktopRules)
ruleExplorer(desktopRules)

##Laptop Rules
laptopRules <- subset(rules.pruned, subset = rhs %pin% "Laptop")
inspect(laptopRules)
plot(laptopRules)
ruleExplorer(laptopRules)

##Dell Desktop Rules##
dellDesktopRules <- subset(rules.pruned, subset = rhs %in% "Dell Desktop")
inspect(dellDesktopRules)
plot(dellDesktopRules)
ruleExplorer(dellDesktopRules)

##iMac Rules##
iMacRulesRHS <- subset(rules.pruned, subset = rhs %in% "iMac")
iMacRulesSuppRHS <- sort(iMacRulesRHS, by = "lift")
inspect(head(iMacRulesSuppRHS))
inspectDT(iMacRulesRHS)
plot(iMacRulesRHS)
ruleExplorer(iMacRulesRHS)

iMacRulesLHS <- subset(rules.pruned, subset = lhs %in% "iMac")
iMacRulesSuppLHS <- sort(iMacRulesLHS, by = "support")
inspect(iMacRulesSuppLHS)
inspectDT(iMacRulesLHS)
plot(iMacRulesLHS)

##
rulesV2 <- apriori(transactionData, parameter = list(supp=0.01, minlen=2, conf=0.5))
rulesV2.sorted <- sort(rulesV2, by="lift")
plot(rulesV2.sorted)
rulesRedunV2 <- is.redundant(rulesV2.sorted)
rulesRedunV2
inspect(rulesV2.sorted)
ruleExplorer(rulesV2.sorted)

rulesV3 <- apriori(transactionData, parameter = list(supp=0.01, minlen=2, conf=0.6))
rulesV3.sorted <- sort(rulesV3, by="lift")
plot(rulesV2.sorted)
inspect(rulesV3.sorted)
ruleExplorer(rulesV3.sorted)

rulesV4 <- apriori (transactionData, parameter = list(supp = 0.001,minlen = 2, maxlen=3, conf = 0.8))
rulesV4.sorted <- sort(rulesV4, by="lift")
rulesV4Redun <- is.redundant(rules.sortedV4)
inspect(rulesV4.sorted)
ruleExplorer(rules.sortedV4)





# keyboardSub <- subset(rules.pruned,  subset = rhs %in% "Keyboard" )
# inspect(keyboardSub)
# keyboardRulesRHS <- apriori(data = transactionData, parameter = list(supp=0.1, conf=0.8), appearance = list(default="lhs",rhs="Keyboard"), control = list(verbose=F))
# keyboardRulesRHSSort <- sort(keyboardRulesRHS, by="confidence", decreasing = TRUE)
# inspect(keyboardRulesRHSSort)
# inspectDT(keyboardRulesRHSSort)
# 
# keyboardRulesLHS <- apriori(data = transactionData, parameter = list(supp=0.01, conf=0.2), appearance = list(default="rhs",lhs="Keyboard"), control = list(verbose=F))
# keyboardRulesLHSSort <- sort(keyboardRulesLHS, by="confidence", decreasing = TRUE)
# inspect(keyboardRulesLHSSort)
# inspectDT(keyboardRulesLHSSort)


###Test Alternative Conf Level###
# aprioriRules50per <- apriori (transactionData, parameter = list(supp = 0.1, conf = 0.5))
# 
# aprioriRules50per
# inspect(aprioriRules50per)
# 
# summary(aprioriRules50per)
# inspect(aprioriRules50per)
# inspect(sample(sort(aprioriRules50per, by = "confidence"),5))
# 
# ?subset
# keyboardRules50per <- subset(aprioriRules50per,  subset = rhs %in% "Keyboard" )
# keyboardRules50per
# 
# inspect(keyboardRules50per)
# 
# is.redundant(aprioriRules50per)
# 
# plot(aprioriRules50per)
# plot(aprioriRules50per[1:26], method="graph", control=list(type="b")) 
# head(quality(aprioriRules50per))
# plot(aprioriRules50per, measure = c("support","lift"),shading = "confidence")
# plot(aprioriRules50per, method = "two-key plot")
# inspectDT(aprioriRules50per)






## 05-Nov-2017
######################## Colloborative filtering 
library("recommenderlab")

###################  Data Prep
setwd("D:\\Recommender\\EDA")
data <- read.csv("rating_final.csv")
#From the data set, we use only, userID, PLaceID and rating
data$food_rating <- NULL
data$service_rating <- NULL

# creating the rating matrix. ie. USERID, ITEMID matrix for the ratings
r <- as(data, "realRatingMatrix")


################### Model - User based 
# Model buildign 
fit <- Recommender(r, "UBCF")

################## Finding similar users 

# using Cosine similarity



################### Predictiong 
rec <- predict(fit, r[10:11], n=3)


# show recommendations
as(rec,'list')


################## Evaluation
# evaluation scheme
e <- evaluationScheme(r,method='split',train=0.9, given=3)

# training model 
train_fit <- Recommender(getData(e,'train'),"UBCF")

# prediction
test_rating <- predict(train_fit, getData(e,"known"), type='ratings')

# error
error <- calcPredictionError(test_rating, getData(e, "unknown"))

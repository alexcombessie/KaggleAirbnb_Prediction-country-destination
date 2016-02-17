################################################################################################################################
######################### KAGGLE AIRBNB: Prediction of country destination for new users ########################################
################################################################################################################################


####################### 1. Workspace and Data Loading ############################################

lib<-c("DMwR","e1071", "nnet", "randomForest", "adabag","xgboost","data.table","Matrix",
       "sqldf","tidyr","readr","stringr","RPostgreSQL","RMySQL","mxnet","h2o")
sapply(lib, require, character.only = TRUE, quietly=T)

#To be modified by each user
dir<-"C:/Users/acombess/Dropbox/ENSAE/Semestre 1/Apprentissage statistique/Kaggle Airbnb"
setwd(dir)

totalclasses<-c("character","Date","character","Date","factor","numeric",rep("factor",10))
df_train<-read.csv("./Kaggle-Data/train_users_2.csv",stringsAsFactors=TRUE,na.strings=c("","-unknown-"),
                   colClasses = totalclasses)
df_test<-read.csv("./Kaggle-Data/test_users.csv",stringsAsFactors=TRUE,na.strings=c("","-unknown-"),
                  colClasses = head(totalclasses,-1))

session<-read.csv("./sessions.csv",na.strings=c(""," ","-unknown-", "NA"))
session["actionall"]<-do.call(paste, c(session[c("action", "action_type","action_detail","device_type")], sep = "."))
session["action"]<-NULL
session["action_type"]<-NULL
session["action_detail"]<-NULL
session["device_type"]<-NULL

df_sessions_group1 <-sqldf('select user_id, actionall, sum(secs_elapsed) from session 
                           where user_id is not null group by user_id, actionall', dbname = tempfile(),drv="SQLite")
names(df_sessions_group1) <- c("id","actionall","secs_elapsed")
df_sessions_pivot1<-spread(df_sessions_group1,actionall, secs_elapsed)
names(df_sessions_pivot1)<-sapply(names(df_sessions_pivot1),function(x) paste("sum",x,sep="_"))
df_sessions_pivot1[is.na(df_sessions_pivot1)]<-0

df_sessions_group2 <-sqldf('select user_id, actionall, count(secs_elapsed) from session 
                           where user_id is not null group by user_id, actionall', dbname = tempfile(),drv="SQLite")
names(df_sessions_group2) <- c("id","actionall","count")
df_sessions_pivot2<-spread(df_sessions_group2,actionall,count)
names(df_sessions_pivot2)<-sapply(names(df_sessions_pivot2),function(x) paste("count",x,sep="_"))
df_sessions_pivot2[is.na(df_sessions_pivot2)]<-0


df_sessions_pivot<-cbind(df_sessions_pivot1,df_sessions_pivot2[,2:ncol(df_sessions_pivot2)])
colnames(df_sessions_pivot)[1]<-"id"

df_train <- merge(df_train, df_sessions_pivot, by = "id", all.x = TRUE)
df_test <- merge(df_test, df_sessions_pivot, by = "id", all.x = TRUE)

rm(df_sessions_group1,df_sessions_group2,df_sessions_pivot1,df_sessions_pivot2,df_train_imputed2)



####################### 2.Data exploration ############################################
summary(df_train)
summary(df_test)



###################### 3. Data preparation #########################################

###### Date feature engineering

date_preparation<-function(x){
  x[,"month_account_created"]<-as.factor(months(x[,"date_account_created"]))
  x[,"weekday_account_created"]<-as.factor(weekdays(x[,"date_account_created"]))
  x[,"day_account_created"]<-as.factor(format(x[,"date_account_created"],"%d"))
  x[,"timestamp_first_active"]<-as.Date(as.POSIXct(x[,"timestamp_first_active"], format="%Y%m%d%H%M%S"))
  x[,"month_first_active"]<-as.factor(months(x[,"timestamp_first_active"]))
  x[,"weekday_first_active"]<-as.factor(weekdays(x[,"timestamp_first_active"]))
  x[,"day_first_active"]<-as.factor(format(x[,"timestamp_first_active"],"%d"))
  x[,"diff_account_active"]<-as.numeric(difftime(x[,"timestamp_first_active"],x[,"date_account_created"],units="hours"))
  return(x)
}

df_train<-date_preparation(df_train)
df_test<-date_preparation(df_test)


###### Agregation fo rare modalities

get_rare_classes<-function(df,tx){
  rare_classes<-list()
  for(r in colnames(df))  {
    if(class(df[,r])=="factor"){
      rare_classes[[r]]<-names(which(summary(df[,r])<tx*sum(!is.na(df[,r]))))
    }
  }
  return(rare_classes)
}

rare_classes<-get_rare_classes(df_train,0.001)

simplify_classes<-function(df,rares_classes){
  for(r in names(rare_classes)){
    if(length(rare_classes[[r]])>1){
      otherclass<-paste(rare_classes[[r]],collapse="+")
      levels(df[,r]) <- c(levels(df[,r]), otherclass)
      df[df[,r] %in% rare_classes[[r]], r]<- otherclass
      df[,r]<-droplevels(df[,r])
    }
  }
  return(df)
}

df_train<-simplify_classes(df_train,rare_classes)
df_test<-simplify_classes(df_test,rare_classes)

harmonize_levels<-function(train,test){
  for(r in colnames(test))  {
    if(class(test[,r])=="factor"){
      test[,r] <- factor(test[,r], levels=levels(train[,r]))
    }
  }
  return(test)
}

df_test<-harmonize_levels(df_train,df_test)

###### Elimination of extreme values of age

extreme_age <- !(df_train[,"age"] > 17 & df_train[,"age"] < 100) & !is.na(df_train[,"age"])
df_train[extreme_age,"age"] <- NA


###### Elimination of non predictive columns

FirstPredictors<-c("gender","age","signup_method","signup_flow","language",
              "affiliate_channel","affiliate_provider","first_affiliate_tracked",
              "signup_app","first_device_type","first_browser","month_account_created",
              "weekday_account_created","month_first_active",
              "weekday_first_active","diff_account_active")

rownames(df_train) <- df_train[,1]
rownames(df_test) <- df_test[,1]

df_train <- df_train[,!names(df_train) %in% c("id","date_account_created","timestamp_first_active","date_first_booking")]
df_test <- df_test[,!names(df_test) %in% c("id","date_account_created","timestamp_first_active","date_first_booking")]


######Imputation of missing value

# CAN BE VERY LONG (30h of computation)

df_train_imputed[,!names(df_train_imputed) %in% FirstPredictors]<-centralImputation(df_train_imputed[,!names(df_train_imputed) %in% FirstPredictors])
df_test_imputed[,!names(df_test_imputed) %in% FirstPredictors]<-centralImputation(df_test_imputed[,!names(df_test_imputed) %in% FirstPredictors])


df_test_imputed[,FirstPredictors]<-knnImputation(df_test_imputed[,FirstPredictors], k=7, 
                                                distData = df_train_imputed[,FirstPredictors])
df_train_imputed[,FirstPredictors]<-knnImputation(df_train_imputed[,FirstPredictors], k=7)

labels_train<-as.numeric(df_train_imputed[,"country_destination"])-1

colnames(df_train_imputed)<-make.names(colnames(df_train_imputed))
colnames(df_test_imputed)<-make.names(colnames(df_test_imputed))

formula_predictors<-formula(paste("country_destination ~",
                                  paste(names(df_train_imputed)[!names(df_train_imputed) %in% c("country_destination")],
                                        collapse="+"),sep=" "))

formula_predictors_notarget<-formula(paste("~",paste(paste(names(df_train_imputed)[!names(df_train_imputed) %in% c("country_destination")],
                                                           collapse="+"),collapse=" + "),sep=" "))

get_top5<-function(df,x){
  paste(colnames(df)[sort(x,decreasing = T,index.return=T)$ix[1:5]],collapse=' ')
}


###################### 4. Machine Learning modeling #########################################

###### WARNING : the code below has been used manually on different versions of the prepared dataset
###### Some models only run on smaller versions of the database with less features from the 'sessions' dataset
##### Adapt it for your own use

fit_naivebayes_imputed<-naiveBayes(formula_predictors_old,data=df_train_imputed)
predicted_naivebayes_imputed<-predict(fit_naivebayes_imputed, df_test_imputed, "raw")
predicted_top5_naivebayes_imputed<-strsplit(apply(predicted_naivebayes_imputed,MARGIN=1,
                                          function(x) get_top5(predicted_naivebayes_imputed,x)),split=" ")

fit_multinomlogit_imputed<-multinom(formula_predictors,data=df_train_imputed, MaxNWts = 8000, 
                                    maxit=800,na.action=na.omit)
predicted_multinomlogit_imputed<-predict(fit_multinomlogit_imputed, df_test_imputed, "probs")
predicted_top5_multinomlogit_imputed<-strsplit(apply(predicted_multinomlogit_imputed,MARGIN=1,
                                             function(x) get_top5(predicted_multinomlogit_imputed,x)),split=" ")

fit_randomforest_imputed<-randomForest(formula_predictors,data=df_train_imputed,na.action = na.omit,
                               importance=TRUE, do.trace=1,keep.forest=TRUE,ntree=300)
predicted_randomforest_imputed<-predict(fit_randomforest_imputed, df_test_imputed,"prob")
predicted_top5_randomforest_imputed<-strsplit(apply(predicted_randomforest_imputed,MARGIN=1,
                                            function(x) get_top5(predicted_randomforest_imputed,x)),split=" ")

class_weights<-log(1/(table(df_train[,"country_destination"])/sum(table(df_train[,"country_destination"]))),10)
fit_randomforest_weighted_imputed<-randomForest(formula_predictors,data=df_train_imputed,na.action = na.omit,
                                                do.trace=10,ntree=200,classwt=class_weights)
predicted_randomforest_weighted_imputed<-predict(fit_randomforest_weighted_imputed, df_test_imputed,"prob")
predicted_top5_randomforest_weighted_imputed<-strsplit(apply(predicted_randomforest_weighted_imputed,MARGIN=1,
                                                     function(x) get_top5(predicted_randomforest_weighted_imputed,x)),split=" ")

fit_nnet_imputed<-nnet(formula_predictors,data=df_train_imputed, size=50, 
               MaxNWts = 6000, maxit=800, na.action=na.omit)
predicted_nnet_imputed<-predict(fit_nnet_imputed, df_test_imputed, "raw")
predicted_top5_nnet_imputed<-strsplit(apply(predicted_nnet_imputed,MARGIN=1,
                                    function(x) get_top5(predicted_nnet_imputed,x)),split=" ")

sparse_matrix_train<- sparse.model.matrix(country_destination ~ ., 
                                          data = df_train_imputed,row.names=F,verbose=T)[,-1]
sparse_matrix_test<- sparse.model.matrix(formula_predictors_notarget, 
                                         data = df_test_imputed,row.names=F,verbose=T)[,-1]

ndcg5 <- function(preds, dtrain) {
  labels <- getinfo(dtrain,"label")
  num.class = 12
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

#Various parameters have been tried manually (I had not enough time for proper cross-validation)
par  <-  list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "ndcg5",
              subsample=0.8,
              max_depth = 20,
              verbose = 10,
              num_class = 12)
fit_boosting_imputed<-xgboost(param=par,data=sparse_matrix_train,label=labels_train,nrounds=500)

predicted_boosting_imputed<-matrix(data=predict(fit_boosting_imputed, sparse_matrix_test),ncol=12,byrow=T)
colnames(predicted_boosting_imputed)<-levels(df_train[,"country_destination"])
predicted_top5_boosting_imputed<-strsplit(apply(predicted_boosting_imputed,MARGIN=1,
                                            function(x) get_top5(predicted_boosting_imputed,x)),split=" ")


###################### 5. Output results #########################################

write.csv(data.frame(id=row.names(df_test)[rep(1:nrow(df_test),each=5)],country=unlist(predicted_top5_naivebayes_imputed)),quote=F,row.names = F,
          file="submission_naivebayes_imputed.csv")
write.csv(data.frame(id=df_test[rep(1:nrow(df_test),each=5),"id"],country=unlist(predicted_top5_multinomlogit_imputed)),quote=F,row.names = F,
          file="submission_multinomlogit_imputed.csv")
write.csv(data.frame(id=df_test[rep(1:nrow(df_test),each=5),"id"],country=unlist(predicted_top5_randomforest_imputed)),quote=F,row.names = F,
          file="submission_rf_imputed.csv")
write.csv(data.frame(id=df_test[rep(1:nrow(df_test),each=5),"id"],country=unlist(predicted_top5_randomforest_weighted_imputed)),quote=F,row.names = F,
          file="submission_rf_weighted_imputed.csv")
write.csv(data.frame(id=df_test[rep(1:nrow(df_test),each=5),"id"],country=unlist(predicted_top5_nnet_imputed)),quote=F,row.names = F,
          file="submission_nnet_imputed.csv")
write.csv(data.frame(id=row.names(df_test_imputed)[rep(1:nrow(df_test_imputed),each=5)],country=unlist(predicted_top5_boosting_imputed)),quote=F,row.names = F,
          file="xgboost_sumcount_sampled.csv")



###################### 6. Model Blending #########################################

lf<-list.files("./best-submissions/")
subm<-vector("list", length(lf))

for (i in 1:length(lf)){
  subm[[i]]<-read.csv(paste("./best-submissions/",lf[i],sep=""),colClasses = c("character","factor"))
  colnames(subm[[i]])<-c("id",lf[i])
}

numberrow<-nrow(subm[[1]])
idsimple<-vector("list", numberrow/5)
country1<-vector("list", numberrow/5)
country2<-vector("list", numberrow/5)
country3<-vector("list", numberrow/5)
country4<-vector("list", numberrow/5)
country5<-vector("list", numberrow/5)
dataframe_pred<-vector("list", numberrow/5)

for (i in 1:length(lf)){
  idsimple[[i]]<-as.character(subm[[i]][seq(1,numberrow,5),"id"])
  country1[[i]]<-as.factor(subm[[i]][seq(1,numberrow,5),2])
  country2[[i]]<-as.factor(subm[[i]][seq(2,numberrow+1,5),2])
  country3[[i]]<-as.factor(subm[[i]][seq(3,numberrow+2,5),2])
  country4[[i]]<-as.factor(subm[[i]][seq(4,numberrow+3,5),2])
  country5[[i]]<-as.factor(subm[[i]][seq(5,numberrow+4,5),2])
  dataframe_pred[[i]]<-data.frame(id=idsimple[[i]],country1=country1[[i]],country2=country2[[i]],
                                  country3=country3[[i]],country4=country4[[i]],country5=country5[[i]])
  colnames(dataframe_pred[[i]])<-c("id",sapply(c("country1","country2","country3","country4","country5"),
                                               function(x) paste(lf[[i]],x,sep=".")))
}

prioritymodel<-1

merged<-merge(merge(merge(merge(dataframe_pred[[1]],dataframe_pred[[2]],by="id"),dataframe_pred[[3]],by="id"),dataframe_pred[[4]],by="id"),
              dataframe_pred[[prioritymodel]],by="id")
rownames(merged)<-merged[,"id"]
merged[,"id"]<-NULL

for(i in 1:5){
  merged["blendcountry1"]<-apply(merged[,seq(1,25,5)],1,function(x) names(which.max(table(x))))
  merged["blendcountry2"]<-apply(merged[,seq(2,26,5)],1,function(x) names(which.max(table(x))))
  merged["blendcountry3"]<-apply(merged[,seq(3,27,5)],1,function(x) names(which.max(table(x))))
  merged["blendcountry4"]<-apply(merged[,seq(4,28,5)],1,function(x) names(which.max(table(x))))
  merged["blendcountry5"]<-apply(merged[,seq(5,29,5)],1,function(x) names(which.max(table(x))))
}

test<-merged[,seq(5,34,5)]
blendedresults<-merged[,26:30]
blendedresults_vector<-as.vector(as.matrix(t(blendedresults)))

write.csv(data.frame(id=row.names(blendedresults)[rep(1:nrow(blendedresults),each=5)],country=blendedresults_vector),
          quote=F,row.names = F, file="blend.csv")



###################### 7. Evaluation #########################################

### WARNING: THESE ARE OLD RESULTS FOR BENCHMARKING PURPOSES. FINAL SUBMISSIONS USED WERE ONLY FROM DIFFERENT VERSIONS OF XGBOOST.
results<-sort(c("Naive Bayes"=0.83874,"Logit Multinomial"=0.85337,"Random Forest"=0.82701,
                "Random Forest +poids"=0.69293,"Reseau de neurones"=0.83352,"Boosting"=0.83958),decreasing=T)
plotresults<-barplot(results,ylim=c(0.66,0.88),
                     xpd=F,cex.names=0.63,cex.axis=0.63)
grid(ny=NULL,nx=1,col="darkgrey")
text(x=plotresults,y=results,labels=results, pos=3,cex=0.8)

### FINAL RESULTS FROM XGBOOST MODELS ON THE PRIVATE LEADERBOARD
results<-sort(c("Blended Model"=0.87258,"XGBoost SUM sessions with sampling"=0.86550,"XGBoost SUMCOUNT sessions without sampling"=0.87062,
                "XGBoost SUM sessions without sampling"=0.87297,"XGBoost SUMCOUNT sessions with sampling"=0.86971),decreasing=T)
plotresults<-barplot(results,ylim=c(0.8,0.9),
                     xpd=F,cex.names=0.63,cex.axis=0.63)
grid(ny=NULL,nx=1,col="darkgrey")
text(x=plotresults,y=results,labels=results, pos=3,cex=0.8)
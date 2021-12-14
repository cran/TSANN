#' @title Time Series Artificial Neural Network
#' @description The best ANN structure for time series data analysis is a demanding need in the present era. This package will find the best-fitted ANN model based on forecasting accuracy. The optimum size of the hidden layers was also determined after determining the number of lags to be included. This package has been developed using the algorithm of Paul and Garai (2021) <doi:10.1007/s00500-021-06087-4>.
#' @param data Time Series Data
#' @param min.size Minimum Size of Hidden Layer
#' @param max.size Maximum Size of Hidden Layer
#' @param split.ratio Training and Testing Split Ratio
#' @return A list containing:
#' \itemize{
#'   \item FinalModel: Best ANN model
#'   \item Trace: Matrix of All Iteration
#'   \item FittedValue: Model Fitted Value
#'   \item PredictedValue: Model Forecast Value of Test Data
#'   \item Train.RMSE: Root Mean Square Error of Train Data
#'   \item Test.RMSE: Root Mean Square Error of Test Data
#' }

#' @import forecast stats gtools utils
#'
#' @export
#'
#' @examples
#' set.seed(16)
#' x<-rnorm(n = 50, mean = 150, sd = 10)
#' Auto.TSANN(x,1,2,0.80)
#' @references
#' Paul, R.K. and Garai, S. (2021). Performance comparison of wavelets-based machine learning technique for forecasting agricultural commodity prices, Soft Computing, 25(20), 12857-12873

Auto.TSANN<- function(data,min.size, max.size, split.ratio)
{
  y<-as.ts(data)
  max.p<-sum(acf(y)$acf>0.05)
  min.p<-1
  train_valid <- as.ts(head(data, round(length(y) * split.ratio)))
  test<- as.ts(tail(y,(length(y) - length(train_valid))))
  train <- as.ts(head(train_valid, round(length(train_valid) * 0.80)))
  valid<-as.ts(head(train_valid, (length(train_valid) - length(train))))
  a <- seq(min.p, max.p, by = 1)
  b <- seq(min.size, max.size, by = 1)
  output <- matrix(nrow=length(a)*length(b), ncol=5)
  res <- vector()
  for(j in 1:length(b)) {


    for(i in 1:length(a)) {
      ann<- forecast::nnetar(train,p=a[i], size=b[j])
      ann.valid<-forecast::nnetar(valid,model=ann)
      ann$model
      ann.train.valid<-forecast::nnetar(train_valid,model=ann)
      fittted_train<-as.ts(ann$fitted)
      fitted_valid<-as.ts(ann.valid$fitted)
      predict_ann<-as.ts(forecast::forecast(ann.train.valid,h=length(test)),data=test)

      valid_rmse<-sqrt(mean(na.omit(((valid -fitted_valid)^2))))
      train_rmse<-sqrt(mean(na.omit(((train -fittted_train)^2))))
      test_rmse<-sqrt(mean(na.omit(((test -predict_ann)^2))))

      output[i+(j-1)*length(a),1]<-a[i]
      output[i+(j-1)*length(a),2]<-b[j]
      output[i+(j-1)*length(a),3]<-train_rmse
      output[i+(j-1)*length(a),5]<-test_rmse
      output[i+(j-1)*length(a),4]<-valid_rmse

    }
  }
  Trace<-round(as.data.frame(output), digits = 4)
  colnames(Trace) <- c('lags','size','train_RMSE', 'valid_RMSE',  'test_RMSE')
  min_row<-Trace[which.min(Trace$test_RMSE),]
  model<- forecast::nnetar(train_valid,p=min_row$lags, size=min_row$size)
  final.model<-model$model
  fitted.value<-round(as.ts(model$fitted),digits = 4)
  predicted.value<-round(as.ts(forecast::forecast(model,h=length(test),data=test)),digits = 4)
  train_RMSE<-round(min_row[,3],digits = 4)
  test_RMSE<-round(min_row[,4],digits = 5)
  Trace.Matrix<-Trace[,c(1,2,5)]
  colnames(Trace.Matrix) <- c('lags','size','Accuracy Metric')
  my.list<-list(FinalModel=final.model, Trace=Trace.Matrix,FittedValue=fitted.value,PredictedValue=predicted.value,Train.RMSE=train_RMSE,Test.RMSE=test_RMSE)
  return(my.list)
}

#Pacotes necessarios
library(randomForest)
library(caret)
library(sp)
library(plyr)
library(xgboost)

mydata <-read.csv("C:/tese_revisao/Banco_2000_2010_27.csv", header = TRUE, sep =";", dec = ".")

str(mydata)

mydata$Codigos <- NULL
mydata$Municipios <- NULL
mydata$TH <-NULL
mydata$TL <-NULL
mydata$TLC <-NULL
mydata$THC <-NULL
mydata$TD <-NULL


str(mydata)

names(mydata)

mydata <- as.data.frame(mydata)


#Separando amostra de treino e valida��o

set.seed(300)
conjunto <- createDataPartition(mydata$TDC,list=FALSE,p=0.7)
Treino <- mydata[conjunto,]
Validacao <- mydata[-conjunto,]


#Numero de amostras para treino
nrow(Treino)

#Numero de amostras para valida��o
nrow(Validacao)


#Random Forest

set.seed(100)
RF <- train(TDC~ .,  data = Treino, method = "rf",
            trControl=trainControl(method = "cv",number=5), 
            metric="Accuracy")
print(RF)
plot(RF)


plot(varImp(RF), top=20)
plot(varImp(RF))
importancia <- varImp(RF, type=2)
plot(importancia, top=10, xlim=c(-5, 105), xlab = "Import�ncia (%)", ylab = "Vari�veis")

##Elaborando o modelo C5.0

library(caret)
set.seed(100)
C50 <- train(TDC~ .,  data = Treino, method = "C5.0",
             trControl=trainControl(method = "cv",number=5), 
             metric="Accuracy")

print(C50)

importancia <- varImp(C50, type=2)
plot(importancia, top=10, xlim=c(-5, 105), xlab = "Import�ncia (%)", ylab = "Vari�veis")


#XGBoost 

library(xgboost)

set.seed(100)
xgBoost <- train(TDC ~ .,  data = Treino, method = "xgbTree",
                 trControl=trainControl(method = "CV", number = 5), metric="Accuracy")

print(xgBoost)

importancia <- varImp (xgBoost, type=2)
plot(importancia, top=10, xlim=c(-5, 105), xlab = "Import�ncia (%)", ylab = "Vari�veis")



##Plot comparando os modelos 
comparacao <- resamples(list(C5.0=C50, RF=RF, XGBoost=xgBoost))

# boxplots das valida��es
bwplot(comparacao)

summary(comparacao)


#Matriz de confusao e estatisticas do modelo com as amostras de valida��o
#Primeiro se aplica o modelo no dado de valida��o
RFpred <- as.factor(predict(RF, Validacao, type="raw"))

#Em seguida � criado uma matrix (table) entre 
#o dado de valida��o e o predito pelo modelo
RF_tabela <- table(Validacao$TDC, RFpred)

#Se calcula a matriz de confus�o com as m�tricas estat�sticas
confusionMatrix(RF_tabela)


#Primeiro se aplica o modelo no dado de valida��o
C50pred <- as.factor(predict(C50, Validacao, type="raw"))

#Em seguida � criado uma matrix (table) entre 
#o dado de valida��o e o predito pelo modelo
C50_tabela <- table(Validacao$TDC, C50pred)

#Se calcula a matriz de confus�o com as m�tricas estat�sticas
confusionMatrix(C50_tabela)



#Matriz de confusao e estatisticas do modelo com as amostras de valida��o
#Primeiro se aplica o modelo no dado de valida��o
xgBoostpred <- as.factor(predict(xgBoost, Validacao, type="raw"))

#Em seguida � criado uma matrix (table) entre 
#o dado de valida��o e o predito pelo modelo
xgBoost_tabela <- table(Validacao$TDC,  xgBoostpred)

#Se calcula a matriz de confus�o com as m�tricas estat�sticas
confusionMatrix(xgBoost_tabela)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


wcat = pd.read_csv("calories_consumed.csv")

wcat.plot (x='Calories_Consumed', y = 'Weight_gained', style = 'o')
plt.title('calories_consumed vs Weight_gained')
plt.xlabel(' Calories_Consumed ')
plt.ylabel(' Weight_gained ')
plt.show()
plt.scatter(wcat.Calories_Consumed,wcat.Weight_gained)


## Model 1 using the actual values 
model1 = LinearRegression()
model1.fit(wcat.Calories_Consumed.values.reshape(-1,1),wcat.Weight_gained)
pred1 = model1.predict(wcat.Calories_Consumed.values.reshape(-1,1))
#print(pred1)
print (" Model 1 Actual and Predicted values ")
df=pd.DataFrame({'Actual': wcat.Weight_gained , 'Predict':pred1})
print(df)
## Adjusted R-Squared value
print(("R-sq : " , model1.score(wcat.Calories_Consumed.values.reshape(-1,1),wcat.Weight_gained)))# 0.6700
rmse1 = np.sqrt(np.mean((pred1-wcat.Weight_gained)**2))
print("RMSE: ", rmse1) # 32.760
print("Co-ef: ", model1.coef_)
print("Intercept:", model1.intercept_)
plt.scatter(wcat.Calories_Consumed, pred1, color = 'gray')
plt.plot(wcat.Calories_Consumed, pred1, color = 'red', linewidth=2)
plt.show()
input()
## LR using OLS
print(" Model 1 OLS ")
model=smf.ols("Weight_gained~Calories_Consumed",data=wcat).fit()
print(model.summary())
input()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000) 
plt.show()
## checking normal distribution for residual
plt.hist(pred1-wcat.Weight_gained)
plt.show()
input()



### Fitting Quadratic Regression 
print("Model 2 : ")
wcat["Calories_sqrd"] = wcat.Calories_Consumed*wcat.Calories_Consumed
model2 = LinearRegression()
print(wcat.columns)
model2.fit(X = wcat.iloc[:,[1,2]],y=wcat.Weight_gained)
pred2 = model2.predict(wcat.iloc[:,[1,2]])
# Adjusted R-Squared value
print("R-sq : " , model2.score(wcat.iloc[:,[1,2]],wcat.Weight_gained))# 0.67791
rmse2 = np.sqrt(np.mean((pred2-wcat.Weight_gained)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model2.coef_)
print("Intercept: " , model2.intercept_)
print("Model 2 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Weight_gained, 'Predict':pred2})
print(df)
plt.scatter(pred2, (pred2-wcat.Weight_gained), color = 'gray')
plt.plot(pred2, (pred2-wcat.Weight_gained), color = 'red', linewidth=2)
plt.show()
input()
print("Model 2 OLS : ")
model=smf.ols("Weight_gained~wcat.iloc[:,[1,2]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred2-wcat.Weight_gained)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred2-wcat.Weight_gained,dist="norm",plot=pylab)
plt.show()



# Let us prepare a model by applying transformation on dependent variable
print("Model 3 : ")
wcat["Weight_gained_sqrt"] = np.sqrt(wcat.Weight_gained)
print(wcat.columns)
input()
model3 = LinearRegression()
model3.fit(X = wcat.iloc[:,[1,2]],y=wcat.Weight_gained_sqrt)
pred3 = model3.predict(wcat.iloc[:,[1,2]])
# Adjusted R-Squared value
print("R-sq : " , model3.score(wcat.iloc[:,[1,2]],wcat.Weight_gained_sqrt))# 0.74051
rmse3 = np.sqrt(np.mean(((pred3)**2-wcat.Weight_gained)**2)) # 32.0507
print("RMSSE : ", rmse3)
print("Co-eff : " , model3.coef_)
print("Intercept " , model3.intercept_)
print(" Model 3 Actual vs Predicted ")
df=pd.DataFrame({'Actual': wcat.Weight_gained_sqrt, 'Predict':pred3})
print(df)
plt.scatter(pred3, (((pred3)**2-wcat.Weight_gained)), color = 'gray')
plt.plot(pred3, (((pred3)**2-wcat.Weight_gained)), color = 'red', linewidth=2)
plt.show()
print("Model 3 OLS : ")
model=smf.ols("Weight_gained_sqrt~wcat.iloc[:,[1,2]]",data=wcat).fit()
print(model.summary())
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000) 
plt.show() 
# checking normal distribution for residuals 
plt.hist((pred3)**2-wcat.Weight_gained)
plt.show()
st.probplot((pred3)**2-wcat.Weight_gained,dist="norm",plot=pylab)
plt.show()




#Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
print(" Model 4 ")
model4 = LinearRegression()
model4.fit(X = wcat.Calories_Consumed.values.reshape(-1,1),y=wcat.Weight_gained_sqrt)
pred4 = model4.predict(wcat.Calories_Consumed.values.reshape(-1,1))
# Adjusted R-Squared value
print("R-sq : " , model4.score(wcat.Calories_Consumed.values.reshape(-1,1),wcat.Weight_gained_sqrt))# 0.7096
rmse4 = np.sqrt(np.mean(((pred4)**2-wcat.Weight_gained)**2)) # 34.165
print("RMSE : " , rmse4)
print("Co-eff : " , model4.coef_)
print("Intercept : " , model4.intercept_)
df=pd.DataFrame({'Actual': wcat.Weight_gained_sqrt , 'Predict':pred4})
print(df)
plt.scatter(wcat.Calories_Consumed, pred4, color = 'gray')
plt.plot(wcat.Calories_Consumed, pred4, color = 'red', linewidth=2)
plt.show()
print("Model 4 OLS : ")
model=smf.ols("Weight_gained_sqrt~wcat.Calories_Consumed",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
st.probplot((pred4)**2-wcat.Weight_gained,dist="norm",plot=pylab)
plt.show()
# Checking normal distribution for residuals 
plt.hist((pred4)**2-wcat.Weight_gained)
plt.show()


#Let us prepae mode using independen variable 

print("Model 5 : ")
wcat["Calories_log"] = np.log(wcat.Calories_Consumed)
model5 = LinearRegression()
print(wcat.columns)
model5.fit(X = wcat.iloc[:,[1,4]],y=wcat.Weight_gained)
pred5 = model5.predict(wcat.iloc[:,[1,4]])
# Adjusted R-Squared value
print("R-sq : " , model5.score(wcat.iloc[:,[1,4]],wcat.Weight_gained))# 0.67791
rmse2 = np.sqrt(np.mean((pred5-wcat.Weight_gained)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model5.coef_)
print("Intercept: " , model5.intercept_)
print("Model 5 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Weight_gained, 'Predict':pred5})
print(df)
plt.scatter(wcat.Calories_Consumed, pred5, color = 'gray')
plt.plot(wcat.Calories_Consumed, pred5, color = 'red', linewidth=2)
plt.show()
print("Model 5 OLS : ")
model=smf.ols("Weight_gained~wcat.iloc[:,[1,4]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred5,(pred5-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred5-wcat.Weight_gained)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred5-wcat.Weight_gained,dist="norm",plot=pylab)
plt.show()






#Let us prepae mode using independent variable 

print("Model 6 : ")
wcat["Calories_log"] = np.log(wcat.Calories_Consumed)
model5 = LinearRegression()
print(wcat.columns)
model5.fit(X = wcat.iloc[:,[1,2,4]],y=wcat.Weight_gained)
pred5 = model5.predict(wcat.iloc[:,[1,2,4]])
# Adjusted R-Squared value
print("R-sq : " , model5.score(wcat.iloc[:,[1,2,4]],wcat.Weight_gained))# 0.67791
rmse2 = np.sqrt(np.mean((pred5-wcat.Weight_gained)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model5.coef_)
print("Intercept: " , model5.intercept_)
print("Model 5 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Weight_gained, 'Predict':pred5})
print(df)
plt.scatter(wcat.Calories_Consumed, pred5, color = 'gray')
plt.plot(wcat.Calories_Consumed, pred5, color = 'red', linewidth=2)
plt.show()
print("Model 5 OLS : ")
model=smf.ols("Weight_gained~wcat.iloc[:,[1,2,4]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred5,(pred5-wcat.Weight_gained),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred5-wcat.Weight_gained)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred5-wcat.Weight_gained,dist="norm",plot=pylab)
plt.show()

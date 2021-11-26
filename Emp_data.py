import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


wcat = pd.read_csv("C:\\Users\\Desktop\\Simple Linear Regression\\emp_data.csv")

wcat.plot (x='Salary_hike', y = 'Churn_out_rate', style = 'o')
plt.title('Salary_hike vs Churn_out_rate')
plt.xlabel(' Salary Hike ')
plt.ylabel(' Churn out rate  ')
plt.show()
plt.scatter(wcat.Salary_hike.values.reshape(-1,1) ,wcat.Churn_out_rate )
input()


## Model 1 using the actual values 
model1 = LinearRegression()
model1.fit(wcat.Salary_hike.values.reshape(-1,1),wcat.Churn_out_rate)
pred1 = model1.predict(wcat.Salary_hike.values.reshape(-1,1))
#print(pred1)
print (" Model 1 Actual and Predicted values ")
df=pd.DataFrame({'Actual': wcat.Churn_out_rate , 'Predict':pred1})
print(df)
## Adjusted R-Squared value
print(("R-sq : " , model1.score(wcat.Salary_hike.values.reshape(-1,1),wcat.Churn_out_rate)))# 0.6700
rmse1 = np.sqrt(np.mean((pred1-wcat.Churn_out_rate)**2))
print("RMSE: ", rmse1) # 32.760
print("Co-ef: ", model1.coef_)
print("Intercept:", model1.intercept_)
plt.scatter(wcat.Salary_hike, pred1, color = 'gray')
plt.plot(wcat.Salary_hike, pred1, color = 'red', linewidth=2)
plt.show()
## LR using OLS
print(" Model 1 OLS ")
model=smf.ols("Churn_out_rate~Salary_hike",data=wcat).fit()
print(model.summary())
input()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-wcat.Churn_out_rate),c="r")
plt.hlines(y=0,xmin=0,xmax=3000) 
plt.show()
## checking normal distribution for residual
plt.hist(pred1-wcat.Churn_out_rate)
plt.show()




### Fitting Quadratic Regression 
print("Model 2 : ")
wcat["Salary_hike_sqrd"] = wcat.Salary_hike*wcat.Salary_hike
print(wcat.columns)
model2 = LinearRegression()
model2.fit(X = wcat.iloc[:,[0,2]],y=wcat.Churn_out_rate)
pred2 = model2.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
print("R-sq : " , model2.score(wcat.iloc[:,[0,2]],wcat.Churn_out_rate))# 0.67791
rmse2 = np.sqrt(np.mean((pred2-wcat.Churn_out_rate)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model2.coef_)
print("Intercept: " , model2.intercept_)
print("Model 2 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Churn_out_rate, 'Predict':pred2})
print(df)
plt.scatter(wcat.Salary_hike_sqrd, pred2, color = 'gray')
plt.plot(wcat.Salary_hike_sqrd, pred2, color = 'red', linewidth=2)
plt.show()
print("Model 2 OLS : ")
model=smf.ols("Churn_out_rate~wcat.iloc[:,[0,2]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-wcat.Churn_out_rate),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred2-wcat.Churn_out_rate)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred2-wcat.Churn_out_rate,dist="norm",plot=pylab)
plt.show()



# Let us prepare a model by applying transformation on dependent variable
print("Model 3 : ")
wcat["Churn_out_rate_sqrd"] = np.sqrt(wcat.Churn_out_rate)
print(wcat.columns)
input()
model3 = LinearRegression()
model3.fit(X = wcat.iloc[:,[0,2]],y=wcat.Churn_out_rate_sqrd)
pred3 = model3.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
print("R-sq : " , model3.score(wcat.iloc[:,[0,2]],wcat.Churn_out_rate_sqrd))# 0.74051
rmse3 = np.sqrt(np.mean(((pred3)**2-wcat.Churn_out_rate_sqrd)**2)) # 32.0507
print("RMSSE : ", rmse3)
print("Co-eff : " , model3.coef_)
print("Intercept " , model3.intercept_)
print(" Model 3 Actual vs Predicted ")
df=pd.DataFrame({'Actual': wcat.Churn_out_rate_sqrd, 'Predict':pred3})
print(df)
plt.scatter(wcat.Salary_hike, pred3, color = 'gray')
plt.plot(wcat.Salary_hike, pred3, color = 'red', linewidth=2)
plt.show()
print("Model 3 OLS : ")
model=smf.ols("wcat.Churn_out_rate_sqrd~wcat.iloc[:,[0,2]]",data=wcat).fit()
print(model.summary())
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-wcat.Churn_out_rate_sqrd),c="r")
plt.hlines(y=0,xmin=0,xmax=4000) 
plt.show()
# checking normal distribution for residuals 
plt.hist((pred3)**2-wcat.Churn_out_rate_sqrd)
plt.show()
st.probplot((pred3)**2-wcat.Churn_out_rate_sqrd,dist="norm",plot=pylab)
plt.show()




#Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
print(" Model 4 ")
model4 = LinearRegression()
model4.fit(X = wcat.Salary_hike.values.reshape(-1,1),y=wcat.Churn_out_rate_sqrd)
pred4 = model4.predict(wcat.Salary_hike.values.reshape(-1,1))
# Adjusted R-Squared value
print("R-sq : " , model4.score(wcat.Salary_hike.values.reshape(-1,1),wcat.Churn_out_rate_sqrd))# 0.7096
rmse4 = np.sqrt(np.mean(((pred4)**2-wcat.Churn_out_rate_sqrd)**2)) # 34.165
print("RMSE : " , rmse4)
print("Co-eff : " , model4.coef_)
print("Intercept : " , model4.intercept_)
df=pd.DataFrame({'Actual': wcat.Churn_out_rate_sqrd , 'Predict':pred4})
print(df)
plt.scatter(wcat.Salary_hike, pred4, color = 'gray')
plt.plot(wcat.Salary_hike, pred4, color = 'red', linewidth=2)
plt.show()
print("Model 4 OLS : ")
model=smf.ols("Churn_out_rate_sqrd~wcat.Salary_hike",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-wcat.Churn_out_rate_sqrd),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
st.probplot((pred4)**2-wcat.Salary_hike,dist="norm",plot=pylab)
plt.show()
# Checking normal distribution for residuals 
plt.hist((pred4)**2-wcat.Churn_out_rate_sqrd)
plt.show()


#Let us prepae mode using independen variable 

print("Model 5 : ")
wcat["Salary_hike_log"] = np.log(wcat.Salary_hike)
model5 = LinearRegression()
print(wcat.columns)
model5.fit(X = wcat.iloc[:,[0,2,4]],y=wcat.Churn_out_rate)
pred5 = model5.predict(wcat.iloc[:,[0,2,4]])
# Adjusted R-Squared value
print("R-sq : " , model5.score(wcat.iloc[:,[0,2,4]],wcat.Churn_out_rate))# 0.67791
rmse2 = np.sqrt(np.mean((pred5-wcat.Churn_out_rate)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model5.coef_)
print("Intercept: " , model5.intercept_)
print("Model 5 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Churn_out_rate, 'Predict':pred5})
print(df)
plt.scatter(wcat.Salary_hike, pred5, color = 'gray')
plt.plot(wcat.Salary_hike, pred5, color = 'red', linewidth=2)
plt.show()
print("Model 5 OLS : ")
model=smf.ols("Churn_out_rate~wcat.iloc[:,[0,2,4]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred5,(pred5-wcat.Churn_out_rate),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred5-wcat.Churn_out_rate)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred5-wcat.Churn_out_rate,dist="norm",plot=pylab)
plt.show()






#Let us prepae mode using independen variable 

print("Model 6 : ")
model6 = LinearRegression()
print(wcat.columns)
input()
model6.fit(X = wcat.iloc[:,[4]],y=wcat.Churn_out_rate)
pred6 = model6.predict(wcat.iloc[:,[4]])
# Adjusted R-Squared value
print("R-sq : " , model6.score(wcat.iloc[:,[4]],wcat.Churn_out_rate))# 0.67791
rmse = np.sqrt(np.mean((pred6-wcat.Churn_out_rate)**2)) # 32.366
print("RMSE: ", rmse)
print("Co-ef : " , model6.coef_)
print("Intercept: " , model6.intercept_)
print("Model 5 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Churn_out_rate, 'Predict':pred6})
print(df)
plt.scatter(wcat.Salary_hike_log, pred6, color = 'gray')
plt.plot(wcat.Salary_hike_log, pred6, color = 'red', linewidth=2)
plt.show()
print("Model 6 OLS : ")
model=smf.ols("Churn_out_rate~wcat.iloc[:,[4]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred6,(pred6-wcat.Churn_out_rate),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred6-wcat.Churn_out_rate)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred6-wcat.Churn_out_rate,dist="norm",plot=pylab)
plt.show()





# print("Model 7 : ")
# wcat["Sorting_Time_log"] = np.log(wcat.Sorting_Time)
# model7 = LinearRegression()
# print(wcat.columns)
# input()
# model7.fit(X = wcat.iloc[:,[4]],y=wcat.Delivery_Time)
# pred7 = model7.predict(wcat.iloc[:,[4]])
# # Adjusted R-Squared value
# print("R-sq : " , model7.score(wcat.iloc[:,[4]],wcat.Delivery_Time))# 0.67791
# rmse = np.sqrt(np.mean((pred7-wcat.Delivery_Time)**2)) # 32.366
# print("RMSE: ", rmse)
# print("Co-ef : " , model7.coef_)
# print("Intercept: " , model7.intercept_)
# print("Model 7 Actual VS Predicted ")
# df=pd.DataFrame({'Actual': wcat.Delivery_Time, 'Predict':pred7})
# print(df)
# plt.scatter(wcat.Sorting_Time_log, pred7, color = 'gray')
# plt.plot(wcat.Sorting_Time_log, pred7, color = 'red', linewidth=2)
# plt.show()
# print("Model 7 OLS : ")
# model=smf.ols("Delivery_Time~wcat.iloc[:,[4]]",data=wcat).fit()
# print(model.summary())
# input ()
# #### Residuals Vs Fitted values
# import matplotlib.pyplot as plt
# plt.scatter(pred7,(pred7-wcat.Delivery_Time),c="r")
# plt.hlines(y=0,xmin=0,xmax=4000)  
# plt.show()
# # Checking normal distribution
# plt.hist(pred7-wcat.Delivery_Time)
# plt.show()
# import pylab
# import scipy.stats as st
# st.probplot(pred7-wcat.Delivery_Time,dist="norm",plot=pylab)
# plt.show()


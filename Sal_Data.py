import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


wcat = pd.read_csv("C:\\Users\\Desktop\\Simple Linear Regression\\Salary_data.csv")

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wcat = norm_func(wcat.iloc[:,:])

print(wcat.iloc[:,:])
input()

wcat.plot (x='YearsExperience', y = 'Salary', style = 'o')
plt.title('YearsExperience vs Salary')
plt.xlabel(' Years Experience ')
plt.ylabel(' Salary ')
plt.show()
# plt.scatter(wcat.YearsExperience,wcat.Salary)
# plt.show()
input()


## Model 1 using the actual values 
model1 = LinearRegression()
model1.fit(wcat.YearsExperience.values.reshape(-1,1),wcat.Salary)
pred1 = model1.predict(wcat.YearsExperience.values.reshape(-1,1))
#print(pred1)
print (" Model 1 Actual and Predicted values ")
df=pd.DataFrame({'Actual': wcat.Salary , 'Predict':pred1})
print(df)
## Adjusted R-Squared value
print(("R-sq : " , model1.score(wcat.YearsExperience.values.reshape(-1,1),wcat.Salary)))# 0.6700
rmse1 = np.sqrt(np.mean((pred1-wcat.Salary)**2))
print("RMSE: ", rmse1) # 32.760
print("Co-ef: ", model1.coef_)
print("Intercept:", model1.intercept_)
plt.scatter(wcat.YearsExperience, pred1, color = 'gray')
plt.plot(wcat.YearsExperience, pred1, color = 'red', linewidth=2)
plt.show()
## LR using OLS
print(" Model 1 OLS ")
model=smf.ols("Salary~YearsExperience",data=wcat).fit()
print(model.summary())
input()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-wcat.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=3000) 
plt.show()
## checking normal distribution for residual
plt.hist(pred1-wcat.Salary)
plt.show()




### Fitting Quadratic Regression 
print("Model 2 : ")
wcat["YearsExperience_sqrd"] = wcat.YearsExperience*wcat.YearsExperience
print(wcat.columns)
model2 = LinearRegression()
model2.fit(X = wcat.iloc[:,[0,2]],y=wcat.Salary)
pred2 = model2.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
print("R-sq : " , model2.score(wcat.iloc[:,[0,2]],wcat.Salary))# 0.67791
rmse2 = np.sqrt(np.mean((pred2-wcat.Salary)**2)) # 32.366
print("RMSE: ", rmse2)
print("Co-ef : " , model2.coef_)
print("Intercept: " , model2.intercept_)
print("Model 2 Actual VS Predicted ")
df=pd.DataFrame({'Actual': wcat.Salary, 'Predict':pred2})
print(df)
plt.scatter(wcat.YearsExperience_sqrd, pred2, color = 'gray')
plt.plot(wcat.YearsExperience_sqrd, pred2, color = 'red', linewidth=2)
plt.show()
print("Model 2 OLS : ")
model=smf.ols("Salary~wcat.iloc[:,[0,2]]",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-wcat.Salary),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
# Checking normal distribution
plt.hist(pred2-wcat.Salary)
plt.show()
import pylab
import scipy.stats as st
st.probplot(pred2-wcat.Salary,dist="norm",plot=pylab)
plt.show()



# Let us prepare a model by applying transformation on dependent variable
print("Model 3 : ")
wcat["Salary_sqrd"] = np.sqrt(wcat.Salary)
print(wcat.columns)
input()
model3 = LinearRegression()
model3.fit(X = wcat.iloc[:,[0,2]],y=wcat.Salary_sqrd)
pred3 = model3.predict(wcat.iloc[:,[0,2]])
# Adjusted R-Squared value
print("R-sq : " , model3.score(wcat.iloc[:,[0,2]],wcat.Salary_sqrd))# 0.74051
rmse3 = np.sqrt(np.mean(((pred3)**2-wcat.Salary_sqrd)**2)) # 32.0507
print("RMSSE : ", rmse3)
print("Co-eff : " , model3.coef_)
print("Intercept " , model3.intercept_)
print(" Model 3 Actual vs Predicted ")
df=pd.DataFrame({'Actual': wcat.Salary_sqrd, 'Predict':pred3})
print(df)
plt.scatter(wcat.Salary_sqrd, pred3, color = 'gray')
plt.plot(wcat.Salary_sqrd, pred3, color = 'red', linewidth=2)
plt.show()
print("Model 3 OLS : ")
model=smf.ols("wcat.Salary_sqrd~wcat.iloc[:,[0,2]]",data=wcat).fit()
print(model.summary())
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred3)**2,((pred3)**2-wcat.Salary_sqrd),c="r")
plt.hlines(y=0,xmin=0,xmax=4000) 
plt.show()
# checking normal distribution for residuals 
plt.hist((pred3)**2-wcat.Salary_sqrd)
plt.show()
st.probplot((pred3)**2-wcat.Salary_sqrd,dist="norm",plot=pylab)
plt.show()




#Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
print(" Model 4 ")
model4 = LinearRegression()
model4.fit(X = wcat.YearsExperience.values.reshape(-1,1),y=wcat.Salary_sqrd)
pred4 = model4.predict(wcat.YearsExperience.values.reshape(-1,1))
# Adjusted R-Squared value
print("R-sq : " , model4.score(wcat.YearsExperience.values.reshape(-1,1),wcat.Salary_sqrd))# 0.7096
rmse4 = np.sqrt(np.mean(((pred4)**2-wcat.Salary_sqrd)**2)) # 34.165
print("RMSE : " , rmse4)
print("Co-eff : " , model4.coef_)
print("Intercept : " , model4.intercept_)
df=pd.DataFrame({'Actual': wcat.Salary_sqrd , 'Predict':pred4})
print(df)
plt.scatter(wcat.Salary_sqrd, pred4, color = 'gray')
plt.plot(wcat.Salary_sqrd, pred4, color = 'red', linewidth=2)
plt.show()
print("Model 4 OLS : ")
model=smf.ols("Salary_sqrd~wcat.YearsExperience",data=wcat).fit()
print(model.summary())
input ()
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((pred4)**2,((pred4)**2-wcat.YearsExperience_sqrd),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
plt.show()
st.probplot((pred4)**2-wcat.YearsExperience_sqrd,dist="norm",plot=pylab)
plt.show()
# Checking normal distribution for residuals 
plt.hist((pred4)**2-wcat.YearsExperience_sqrd)
plt.show()




# Using scikitlearn we will use logistical regression model

# Load in data from csv file that was downloaded from kaggle
# https://www.kaggle.com/washingtonpost/police-shootings
min_columns = ['manner_of_death', 'armed', 'age', 'gender', 'race', 'city', 'state', 'signs_of_mental_illness', 'threat_level','flee', 'body_camera']
removeList = ['manner_of_death', 'armed', 'gender', 'race', 'city', 'state', 'signs_of_mental_illness', 'threat_level','flee', 'body_camera']
dataFrame = pd.read_csv("database.csv", names=min_columns, skiprows=1)


# There is text data in our CSV so we will have to use pandas dummy variables to address this issue and merge the dummies with the dataframe
dummiesList = [dataFrame]
dummiesList.append(pd.get_dummies(dataFrame.manner_of_death))
dummiesList.append(pd.get_dummies(dataFrame.armed))
dummiesList.append(pd.get_dummies(dataFrame.gender))
dummiesList.append(pd.get_dummies(dataFrame.race))
dummiesList.append(pd.get_dummies(dataFrame.city))
dummiesList.append(pd.get_dummies(dataFrame.state))
dummiesList.append(pd.get_dummies(dataFrame.signs_of_mental_illness))
dummiesList.append(pd.get_dummies(dataFrame.threat_level))
dummiesList.append(pd.get_dummies(dataFrame.flee))
dummiesList.append(pd.get_dummies(dataFrame.body_camera))

# Now we can merge the Lists
encodedDF = pd.concat(dummiesList, axis='columns')
encodedDF = encodedDF.drop(removeList, axis='columns')
encodedDF.age = encodedDF[['age']].fillna(0)


# The data is clean
features = encodedDF.columns[1:]

X = encodedDF[features]
Y = encodedDF.age

#======================================================================
#======================================================================
#======================================================================
#======================================================================


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import array
model = LogisticRegression(max_iter=10000)


#Split the data!
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.15,random_state=0)
#Train with the data!
model.fit(Xtrain,Ytrain)
#Test the Model
predictions = model.predict(Xtest)
print(predictions)

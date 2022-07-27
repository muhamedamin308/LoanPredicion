import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn import svm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Read CSV file
df = pd.read_csv('/Users/mohammed/Desktop/ai_project/loan_data.csv')


# Drop All Null Values
df.dropna(inplace=True)


# Columns Logarithm : That helps to increase your model accuracy and prediction
df['ApplicantIncome_Log'] = np.log(df['ApplicantIncome'])
df['LoanAmount_Log'] = np.log(df['LoanAmount'])
df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])


# Drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Loan_ID']
df = df.drop(columns=cols)

# Convert String Values
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status', 'Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# A , B : Data and Target
a = df.drop(columns=['Loan_Status'], axis=1)  # Data
b = df['Loan_Status']  # Target


# Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3, random_state=53)
'''
 X_train = 0.70 from our Data at a  
 X_test = 0.30 form our Data at a
 Y_train = 0.70 from our Target at b
 Y_test = 0.30 from our Target at b
'''


# SVM
# best line or decision boundary that can segregate n-dimensional space
# into classes so that we can easily put the new data point in the correct category in the future
# Just line can segregate the data into groups
print("SVM Classifier : ", end='\n')
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, Y_train)  # train
y_predict_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(Y_test, y_predict_svm)
SVM_report = classification_report(Y_test, y_predict_svm)
SVM_confusion = confusion_matrix(Y_test, y_predict_svm)
print("SVM Accuracy: ", accuracy_svm, end='\n')
print('SVM Accuracy in percentage : ', int(accuracy_svm * 100), '%', end='\n')
print("SVM Mean Square Error: ", mean_squared_error(Y_test, y_predict_svm), end='\n')
print("SVM Classification Report:\n ", SVM_report)
print("SVM Confusion Matrix:\n", SVM_confusion, end='\n\n\n')


# Logistic Regression
# Helps with data which trade with discrete values and only zeros and ones
print("Logistic Regression Classifier : ", end='\n')
LR_model = LogisticRegression(solver='liblinear')
LR_model.fit(a, b)
y_predict_LR = LR_model.predict(a)
accuracy_LR = accuracy_score(b, y_predict_LR)
LR_report = classification_report(b, y_predict_LR)
LR_confusion = confusion_matrix(b, y_predict_LR)
print('LR Accuracy : ', accuracy_LR, end='\n')
print('LR Accuracy in percentage : ', int(accuracy_LR * 100), '%', end='\n')
print("LR Mean Square Error: ", mean_squared_error(b, y_predict_LR), end='\n')
print("LR Classification Report:\n ", LR_report)
'''
precision : true prediction / all predictions
recall : all predictions / all sample count
f1-score : just a calculated value : 2*precision*recall / precision+recall
support : all actual samples selected : 65 zero and the 120 are ones
'''
print("LR Confusion Matrix:\n", LR_confusion, end='\n\n\n')


# Decision Tree ID3
# See the best Tree or best way to Classifier the Dataset into Decision tree
print("Decision Tree ID3 Classifier : ", end='\n')
ID3_model = DecisionTreeClassifier(max_depth=1)
ID3_model.fit(X_train, Y_train)
y_predict_DR = ID3_model.predict(X_test)
ID3_Accuracy = accuracy_score(Y_test, y_predict_DR)
ID3_report = classification_report(Y_test, y_predict_DR)
ID3_confusion = confusion_matrix(Y_test, y_predict_DR)
print("DR ID3 Accuracy: ", ID3_Accuracy, end='\n')
print('DR ID3 Accuracy in percentage : ', int(ID3_Accuracy * 100), '%', end='\n')
print("DR Mean Square Error: ", mean_squared_error(Y_test, y_predict_DR), end='\n')
print("DR ID3 Classification Report:\n ", ID3_report)
print("DR ID3 Confusion Matrix:\n", ID3_confusion, end='\n\n\n')


# Random Forest Classifier
# Show all Decision Trees that can predict
print("Random Forest Classifier : ", end='\n')
RFC_model = RandomForestClassifier(n_estimators=50)
RFC_model.fit(X_train, Y_train)
y_predict_RFC = RFC_model.predict(X_test)
RFC_report = classification_report(Y_test, y_predict_RFC)
RFC_confusion = confusion_matrix(Y_test, y_predict_RFC)
print("RF Score: ", RFC_model.score(X_train, Y_train), end='\n')
print("RF Accuracy: ", accuracy_score(Y_test, y_predict_RFC), end='\n')
print("RF Accuracy in percentage : ", int(accuracy_score(Y_test, y_predict_RFC) * 100), '%', end='\n')
print("RF Mean Square Error: ", mean_squared_error(Y_test, y_predict_RFC), end='\n')
print("RF Classification Report:\n ", RFC_report)
print("RF Confusion Matrix:\n", RFC_confusion, end='\n\n\n')


# Extra Trees Classifier
# All Samples are unique in nature : SquareRoot of (All number of features still exist)
# Select Random Decisions Trees for Classifier
print("Extra Trees Classifier : ", end='\n')
ETC_model = ExtraTreesClassifier(n_estimators=50)
ETC_model.fit(X_train, Y_train)
y_predict_ETC = ETC_model.predict(X_test)
ETC_report = classification_report(Y_test, y_predict_ETC)
ETC_confusion = confusion_matrix(Y_test, y_predict_ETC)
print("ETC Score: ", ETC_model.score(X_train, Y_train), end='\n')
print("ETC Accuracy: ", accuracy_score(Y_test, y_predict_ETC), end='\n')
print("ETC Accuracy in percentage : ", int(accuracy_score(Y_test, y_predict_ETC) * 100), '%', end='\n')
print("ETC Mean Square Error: ", mean_squared_error(Y_test, y_predict_ETC), end='\n')
print("ETC Classification Report:\n ", ETC_report)
print("ETC Confusion Matrix:\n", ETC_confusion)

import pandas as pd
df = pd.read_csv('data/train.csv')
print(df.head(10))
print(df['Embarked'].unique())
print(df['Pclass'].unique())

num = {}
prob = {}
for index,row in df.iterrows():
    type = str(row['Pclass'])+str(row['Embarked'])+ str(row['Sex'])
    if(type in num):
        num[type]+=1
        if(row['Survived']==1):
            prob[type]+=1

    else:
        num[type]=1
        if(row['Survived']==1):
            prob[type]=1.0
        else:
            prob[type]=0.0

for i in prob.keys():
    prob[i]=prob[i]/num[i]

df2 = pd.read_csv('data/test.csv')
a = [["PassengerId","Survived"]]
for i,row in df2.iterrows():
    type = str(row['Pclass'])+str(row['Embarked'])+ str(row['Sex'])
    if(type in prob):
        if(prob[type]>0.5):
            a+=[[row['PassengerId'],1]]
        else:
            a+=[[row['PassengerId'],0]]
    else:
        a+=[[row['PassengerId'],1]]

pd.DataFrame(a[1:], columns=a[0]).to_csv('data/fin.csv', index=False)

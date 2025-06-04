import pandas as pd
df = pd.read_csv('data/train.csv')
print(df.head(10))
print(df['Embarked'].unique())
print(df['Pclass'].unique())

numgen = {}
probgen = {}

num = {}
prob = {}
for index,row in df.iterrows():
    type = str(row['Pclass'])+str(row['Embarked'])+ str(row['Sex'])
    type2 =  str(row['Pclass'])+str(row['Embarked'])
    if(type in num):
        num[type]+=1
        numgen[type2]+=1
        if(row['Survived']==1):
            prob[type]+=1
            probgen[type2]+=1

    else:
        num[type]=1
        numgen[type2]=1

        if(row['Survived']==1):
            prob[type]=1.0
            probgen[type2]=1.0

        else:
            prob[type]=0.0
            probgen[type2]=0.0


l = len(prob.keys())
templ = len(probgen.keys())
for i in prob.keys():
    prob[i]=prob[i]/(num[i]+2*l)

for i in probgen.keys():
    probgen[i]=probgen[i]/(numgen[i]+2*templ)



df2 = pd.read_csv('data/test.csv')
a = [["PassengerId","Survived"]]
for i,row in df2.iterrows():
    type = str(row['Pclass'])+str(row['Embarked'])+ str(row['Sex'])
    if(type in prob):
        if((prob[type])>0.5):
            a+=[[row['PassengerId'],1]]
        else:
            a+=[[row['PassengerId'],0]]
    else:
        type = str(row['Pclass'])+str(row['Embarked'])
        if(probgen[type]/(numgen[i]+2*templ)>0.5):
            a+=[[row['PassengerId'],1]]
        else:
            a+=[[row['PassengerId'],0]]


pd.DataFrame(a[1:], columns=a[0]).to_csv('data/fin.csv', index=False)

survived = 0
didnt = 0
for index, row in df.iterrows():
    if(row['Survived']==1):
        survived+=1
    else:
        didnt+=1
print(survived)
print(didnt)
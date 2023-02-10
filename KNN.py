#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# In[71]:


F=pd.read_csv('C:\\Users\\82102\\Desktop\\Test\\income\\final1612.csv',encoding='euc-kr')
F

# In[72]:


F['class']=0
F.drop(['Unnamed: 14','Unnamed: 15'],axis=1,inplace=True)

# In[73]:


F

# In[74]:


F.dropna(axis=0,inplace=True)
F

# In[77]:


g=F[F['평균처분가능소득']<600].index

# In[78]:


F.drop(g,inplace=True)
F

# In[26]:


F.dropna(axis=0,inplace=True)
F

# In[27]:


v=F['평균처분가능소득']

# In[61]:


a=18000
b=10500
c=6500
d=3800
e=1900

# In[62]:


def get_score(v):
    if v >a:
        score = 'A'
    elif (v <=a) & (v >b):
        score = 'B'
    elif (v <=b) & (v >c):
        score = 'C'
    elif (v <=c) & (v >d):
        score = 'D'
    elif (v <=d) & (v >e):
        score = 'E'
    else:
        score = 'F'
    return score


F["class"] = F["평균처분가능소득"].apply(lambda v: get_score(v))
F

# ##연령이랑 처분가능소득으로 했을떄
# #### F.loc[(F['연령']==20)|(F['연령']==30)&(F['평균처분가능소득']>c),'class']='A'
# #### F.loc[(F['연령']==20)|(F['연령']==30)&(F['평균처분가능소득']<=c),'class']='B'
# #### F.loc[(F['연령']==40)|(F['연령']==50)|(F['연령']==60)&(F['평균처분가능소득']>c),'class']='A'
# #### F.loc[(F['연령']==40)|(F['연령']==50)|(F['연령']==60)&(F['평균처분가능소득']<=c),'class']='C'
# #### F.loc[(F['연령']==70)|(F['연령']==80)|(F['연령']==90)&(F['평균처분가능소득']>c),'class']='B'
# #### F.loc[(F['연령']==70)|(F['연령']==80)|(F['연령']==90)&(F['평균처분가능소득']<=c),'class']='C'
# #### F.loc[(F['연령']==0)|(F['연령']==15)&(F['평균처분가능소득']>=c),'class']='B'
# #### F.loc[(F['연령']==0)|(F['연령']==15)&(F['평균처분가능소득']<=c),'class']='B'
# #### F 

# In[63]:


F['class'][13000]

# In[64]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(F1, test_size=0.2)

#학습 데이터, 테스트 데이터 개수 확인
print(train.shape[0])
print(test.shape[0])

# In[65]:


from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import cross_val_score # k-fold 교차검증

max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, 41): #홀수
  k_list.append(i)

cross_validation_scores = [] # 각 k의 검증 결과 점수들
x_train = train[['소득대비소비','평균소득','평균소비']]
y_train = train[['class']]

for k in k_list:
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10,
                           scoring='accuracy')
  cross_validation_scores.append(scores.mean())

cross_validation_scores
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()

# In[66]:


k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print("The best number of k : " + str(k))

# In[67]:


from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=k)

x_train = train[['소득대비소비','평균소득','평균소비']]
y_train = train[['class']]

#knn 모델 학습
knn.fit(x_train, y_train.values.ravel())

x_test = test[['소득대비소비','평균소득','평균소비']]
y_test = test[['class']]

#테스트 시작
pred = knn.predict(x_test)

#모델 예측 정확도(accuracy) 출력
print("accuracy : " + str(accuracy_score(y_test.values.ravel(), pred)))

# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

# In[ ]:




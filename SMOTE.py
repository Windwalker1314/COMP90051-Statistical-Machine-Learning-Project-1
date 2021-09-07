import pandas as pd

from imblearn.over_sampling import SMOTE

###Idea from
###https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/


### Default sample: 46% accuracy
### to 200: ~50% accuracy (test set), dev acc around 80
### to 100: ~48.5% accuracy (test set), dev acc around 75


train = pd.read_csv('output_train.csv', header = None)


x = train.iloc[:, :-1]
y = train.iloc[:, -1]


over_sampler = SMOTE(random_state = 42)

new_x, new_y = over_sampler.fit_resample(x,y)

new_x.iloc[:,0] = [i for i in range(1, len(new_x)+1)]

new_x.iloc[:,0]

new_x['y'] = new_y

count = train.groupby(by = [961]).count()


for i in range(1,10):

    strategy = {}
    
    for j in range(1,50):
        if count[1][j] < i*50:
            strategy[j] = i*50
        else:
            strategy[j] = count[1][j]
                
    
    over_sampler = SMOTE(random_state = 42, sampling_strategy = strategy)
    
    new_x, new_y = over_sampler.fit_resample(x,y)
    
    new_x.iloc[:,0] = [i for i in range(1, len(new_x)+1)]
    
    new_x.iloc[:,0]
    
    
    new_x['y'] = new_y
    
    
    new_x.to_csv("au0-"+ str(i)+ "_train.csv" , header = None, index = False)




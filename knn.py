
#x,y=train data
#Regression
class KNN:
    def __init__(self):
        #declare instance variables
        self.x =None
        self.y =None
        
        
    def train(self,x,y):
        self.x =x
        self.y =y
        
 
    def predict(self,x,k) :
        distance_label =[
            (self.distance(x,train_point),train_label)
            for train_point,train_label in zip(self.x,self.y)]
        
        neighbors =sorted(distance_label)[:k]
        
        return sum(label for _,label in neighbors /k)
    
    def distance(point_1,point_2):
    return ((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)**0.5


#Classification
from Counter  import Counter
class KNN:
    def __init__(self):
        #declare instance variables
        self.x =None
        self.y =None
        
        
    def train(self,x,y):
        self.x =x
        self.y =y  
    def predict(self,x,k) :
        distance_label =[
            (self.distance(x,train_point),train_label)
            for train_point,train_label in zip(self.x,self.y)]
        
        neighbors =sorted(distance_label)[:k]
        neighbors_labels =[label for dist,label in neighbors]
        
        
        return Counter(neighbors_labels).most_common()[0][0]


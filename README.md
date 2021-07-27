# ML-Algorithm-ImplementationFromScratch

## K-Means Implementation From Scratch

```python

# K Means to find K clusters from unlabeled data /unsupervised learning
#two dimensional points
data =[(x_0,y_0),...,(x_n,y_n)]
#output
labels =[c_0,...,c_n]

def main(data,k):
    centroids  = initialize_centroids(data,k)
    
    while True:
        old_centroid =centroids
        labels =get_labels(data,centroids)
        centroids = update_centroids(data,labels,k)
    
        if should_stop(old_centroids,centroids):
            break
                
    return labels 


def initialize_centroids(data,k):
    x_min =y_min=float("inf")
    x_max =y_max=float("-inf")
    
    for point in data:
        x_min =min(point[0],x_min)
        x_max =max(point[0],x_max)
        y_min =min(point[0],y_min)
        y_max =max(point[0],y_max)
    
    centroids =[]
    
    for i in range(k):
        centroids.append([random_sample(x_min,x_max),random_sample(y_min,y_max =max(point[0],y_max)
)])
        
    return centroids

#generate random sample between low and high instead 0 or 1
def random_sample(low,high):
    return low (high-low) * random.random()

def initialize_centroids_bad(x,k):
    return x[:k]


def get_labels(data,centroids):
    labels =[]
    for point in data:
        min_dist =float("inf")
        label =None
        for i, centroid in enumerate(centroids):
            new_dist =get_distance(point,centroid)
            if min_dist > new_dist:
                min_dist =new_dist
                label=i 
                
        labels.append(label)
    return labels

## Euclidean Distance formula
def get_distance(point_1,point_2):
    return ((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)**0.5


## Update Centroids 
def update_centroids(points,labels,k):
    new_centroids =[(0,0) for i in range(k)]
    counts =[0] * k
    
    for point,label in zip(points,labels):
        new_centroids[label[0]] += point[0]
        new_centroids[label[1]] += point[1]
        counts[label] +=1
    
    for i,(x,y) in enumerate(new_centroids):
        new_centroids[i] = (x / counts[i], y/counts[i])
    return new_centroids

def should_stop(old_centroids,new_centroids,threshold=1e-5):
    total_movement =0
    for old_point,new_point in zip(old_centroids,new_centroids):
        total_movement +=get_distance(old_point,new_point)
    return total_movement < threshold
```

##  KNN Implementation From Scratch


```python

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

```

 ##  Linear Regression Implementation From Scratch
 
 ```python 
 def linear_regression(x,y,iteration=100,learning_rate=0.01):
    n,m =len(x[0],len(x))
    beta_0,beta_other,initialize_params(n)
    for _ in range(iterations):
        gradient_beta_0,gradient_beta_other =compute_gradient(x,y,beta_0,beta_other,n,m)
        beta_0,beta_other =update_params(beta_0,beta_other,gradient_beta_0,gradient_beta_other,learning_rate)
    return beta_0,beta_other



def initiliaze_params(dimension) :
    beta_0 =0
    beta_other =[random.random() for _ in range(dimension)]
    
    return beta_0,beta_other

def compute_gradient(x,y,beta_0,beta_other,dimension,m):
    gradient_beta_0 =0
    gradient_beta_other =[0]*dimension
    
    for i in range(m):
        y_i_hat =sum(x[i][j] *beta_other[j] for j in range(dimension))+beta_0
        derror_dy =2 *(y[1]-y_i_hat)
        for j in range(dimension):
            gradient_beta_other[j] +=derror_dy*x[i][j]/m
            
        gradient_beta_0 +=derror_dy /m
        
    return gradient_beta_0,gradient_beta_other

def update_params(beta_0,beta_other,gradient_beta_0,gradient_beta_other,learning_rate):
    beta_0 += gradient_beta_0 * learning_rate
    for i in range(len(beta_1)):
        beta_other[i] +=(gradient_beta_other[i]*learning_rate)
        
    return beta_0,beta_other
 
 ```
 
 
 ##  Logistic Regression Implementation From Scratch
 
 ```python 
 
 def logistic_regression(x,y,iterations=100,learning_rate=0.01):
    m,n =len(x),len(x[0])
    beta_0,beta_other =initialize_params(n)
    
    for _ in range(iterations):
        gradient_beta_0,gradient_beta_other =(compute_gradients(x,y,beta_0,beta_other,m,n,50))
        beta_0,beta_other =update_params(beta_0,beta_other,gradient_beta_o,gradient_beta_other,learning_rate)
    
    return beta_0,beta_other

def initiliaze_params(dimension) :
    beta_0 =0
    beta_other =[random.random() for _ in range(dimension)]
    
    return beta_0,beta_other

def compute_gradient(x,y,beta_0,beta_other,m,n):
    gradient_beta_0 =0
    gradient_beta_other =[0]*n
    
    for i in enumerate(x):
        pred = logistic_function(point_beta_0,beta_other)
        
        for j,feature in enumerate(point):
            gradient_beta_other[j] +=(pred-y[i])*feature/m
            gradient_beta_0 +=(pred-y[i])/m
        
    return gradient_beta_0,gradient_beta_other

def logistic_function(point,beta_0,beta_other):
    return 1 /()1 +np.exp(-(beta_0 + point.dot(beta_other)))


def update_params(beta_0,beta_other,gradient_beta_0,gradient_beta_other,learning_rate):
    beta_0 -=gradient_beta_0 * learning_rate
    
    for i in range(len(beta_other)):
        beta_other[i] -=(gradient_beta_other[i] *learning_rate)
    
    return beta_0,beta_other



# For Minibatch Gradient_descent for large dataset

def compute_gradient_minibatch(x,y,beta_0,beta_other,m,n,batch_size):
    gradient_beta_0 =0
    gradient_beta_other =[0] * n
    
    for _ in range(batch_size):
        i =random.randint(0,m-1)
        point =x[i]
        pred =logistic_function(point_beta_0,beta_other)
        for j,feature in enumerate(point):
            gradient_beta_other[j] +=(pred-y[i])*feature/batch_size
            gradient_beta_0 +=(pred-y[i])/batch_size
            
    return gradient_beta_0,gradient_beta_other

 ```

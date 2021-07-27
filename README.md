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

import numpy as np
import random
import matplotlib.pyplot as plt

class Generator:
    def __init__(self):
        pass

    def __call__(self):
        pass

class GaussianGenerator(Generator):
    def __init__(self, mean, std, dimension=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dimension = dimension
        if dimension is None:
            self.dimension = self.mean.shape[0]
    
    def __call__(self):
        return np.random.normal(self.mean,self.std,self.dimension)

class Discrete1DUniformGenerator(Generator):
    def __init__(self,n):
        super().__init__()
        self.n = n
    
    def __call__(self):
        return random.randint(0,self.n-1)
    
class UniformGenerator(Generator):
    def __init__(self,a,b):
        super().__init__()
        self.a = a 
        self.b = b
    
    def __call__(self):
        return (self.a-self.b)*np.random.random(len(self.a))+self.b


class CustomGenerator(Generator):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __call__(self):
        return self.f()

class SumGenerator(Generator):
    def __init__(self,generators):
        super().__init__()
        self.generators = generators

    def __call__(self):
        return sum([generator() for generator in self.generators])
    
class UnionGenerator(Generator):
    def __init__(self,generators,probas=None):
        if probas is None:
            probas = np.ones(len(generators))/len(generators)
        assert abs(sum(probas)-1)<1e-3
        self.probas = probas
        self.generators = generators

    def __call__(self):
        u = np.random.random()
        acc=0
        for (i,p_i) in enumerate(self.probas):
            acc+=p_i
            if acc>u:
                return self.generators[i]()
        return self.generators[-1]()


class DataGenerators:
    def __init__(self,generators,class_generator=None):
        self.generators = generators
        if class_generator is None:
            self.class_generator = Discrete1DUniformGenerator(len(generators))
        else:
            self.class_generator = class_generator
        self.dimension = len(self.generators[0]())

    @property
    def nb_class(self):
        return len(self.generators)

    def generate_data(self,quantity):
        X = np.zeros((quantity,self.dimension),float)
        y = np.zeros((quantity),int)
        for i in range(quantity):
            y[i] = self.class_generator()
            X[i] = self.generators[y[i]]()
        return X, y

def gauss_generator(dimension=2,nb_class=2,std=1):
    mean = np.zeros((nb_class,dimension))
    generators = []
    if nb_class==2:
        mean[0][0] = 1
        mean[1][0] = -1
    else:    
        for i in range(nb_class):
            mean[i][i] = 1
    for i in range(nb_class):
        generators.append(GaussianGenerator(mean[i],std))
    class_generator = Discrete1DUniformGenerator(nb_class)
    return DataGenerators(generators,class_generator)

def unit_sphere_test(dimension):
    x = np.random.normal(0,1,dimension)
    norm = np.sqrt(np.dot(x,x))
    x = x/norm
    return x


def test():
    generators = []
    generators.append(GaussianGenerator(np.zeros(2),0.3))
    generators.append(UniformGenerator(np.array([-3,-3]),np.array([-1,-1])))
    f = lambda :unit_sphere_test(2)
    generators.append(SumGenerator([CustomGenerator(f),GaussianGenerator(np.zeros(2),0.1)]))
    generators.append(UnionGenerator([GaussianGenerator(np.array([2,-2]),0.3),GaussianGenerator(np.array([3,-3]),0.2)],[0.7,0.3]))
    class_generator = Discrete1DUniformGenerator(len(generators))
    return DataGenerators(generators,class_generator)

def display_classes(X,y):
    color_names = ["red","blue","green","orange"]
    colors = [color_names[i] for i in y]

    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()



if __name__ == "__main__":

    data_generator = test()

    X,y  = data_generator.generate_data(1000)

    display_classes(X,y)


import numpy as np
import random
import matplotlib.pyplot as plt

class Generator:
    def __init__(self):
        pass

    def __call__(self,t=None):
        pass

    def update(self,t):
        pass

class GaussianGenerator(Generator):
    def __init__(self, mean=None, std=None, dimension=None, updator=None, t=0):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dimension = dimension
        self.updator = updator
        # initialize with updator at time=t
        self.update(t)
        # detect dimension using shape of mean
        if dimension is None:
            self.dimension = self.mean.shape[0]
    
    def __call__(self, t=None):
        if t is not None:
            self.update(t)
        return np.random.normal(self.mean,self.std,self.dimension)
    
    def update(self,t):
        if self.updator is not None:
            self.mean, self.std = self.updator(t)

class Discrete1DUniformGenerator(Generator):
    def __init__(self,n):
        super().__init__()
        self.n = n
    
    def __call__(self,t=None):
        return random.randint(0,self.n-1)

    def update(self,t):
        pass
    
class UniformGenerator(Generator):
    def __init__(self,a,b,updator=None,t=0):
        super().__init__()
        self.a = a 
        self.b = b
        self.updator = updator
        # initialize with updator at time=t
        self.update(t)
    
    def __call__(self, t=None):
        if t is not None:
            self.update(t)
        return (self.a-self.b)*np.random.random(len(self.a))+self.b

    def update(self,t):
        if self.updator is not None:
            a, b = updator(t)

class CustomGenerator(Generator):
    def __init__(self, f, time_dependant=False):
        super().__init__()
        self.f = f
        self.time_dependant = time_dependant

    def __call__(self,t=None):
        if self.time_dependant:
            return self.f(t)
        else:
            return self.f()

    def update(self,t):
        pass

class SumGenerator(Generator):
    def __init__(self,generators):
        super().__init__()
        self.generators = generators

    def __call__(self,t=None):
        return sum([generator(t) for generator in self.generators])

    def update(self,t):
        for generator in self.generators:
            generator.update(t)

class UnionGenerator(Generator):
    def __init__(self,generators,probas=None):
        if probas is None:
            probas = np.ones(len(generators))/len(generators)
        assert abs(sum(probas)-1)<1e-3 #assert that the sum of probas is 1
        self.probas = probas
        self.generators = generators

    def __call__(self,t=None):
        u = np.random.random()
        acc=0
        for (i,p_i) in enumerate(self.probas):
            acc+=p_i
            if acc>u:
                return self.generators[i]()
        return self.generators[-1]()
    
    def update(self,t):
        for generator in self.generators:
            generator.update(t)


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

    def generate_data(self,quantity, t=None):
        X = np.zeros((quantity,self.dimension),float)
        y = np.zeros((quantity),int)
        if t is not None:
            self.update(t)
        for i in range(quantity):
            y[i] = self.class_generator()
            X[i] = self.generators[y[i]]()
        return X, y

    def update(self,t):
        for generator in self.generators:
            generator.update(t)

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

def two_gaussians(dim,std):
    target = np.zeros(dim)
    target[0] = 1
    lst_gen = []
    lst_gen.append(GaussianGenerator(target,std))
    lst_gen.append(GaussianGenerator(-target,std))
    return DataGenerators(lst_gen)

def not_convex(dim,noise=0.1):
    generators = []
    center_std = 0.5
    generators.append(GaussianGenerator(np.zeros(dim),center_std))
    f = lambda :unit_sphere_test(dim)
    generators.append(SumGenerator([CustomGenerator(f),GaussianGenerator(np.zeros(dim),noise)]))
    return DataGenerators(generators)

def display_classes(X,y):
    color_names = "rgbcmykw"
    colors = [color_names[i] for i in y]

    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()



if __name__ == "__main__":

    # data_generator = test()

    # X,y  = data_generator.generate_data(1000)

    # display_classes(X,y)

    f1 = lambda t: (np.array([0,t]),1)
    f2 = lambda t: (np.array([0,-t]),1)
    gens = []
    gens.append(GaussianGenerator(updator=f1))
    gens.append(GaussianGenerator(updator=f2))

    data_generator = DataGenerators(gens)

    X,y  = data_generator.generate_data(100)
    display_classes(X,y)
    X,y  = data_generator.generate_data(100,t=10)
    display_classes(X,y)

    


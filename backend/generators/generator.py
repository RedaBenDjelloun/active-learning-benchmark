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
    
    def __call__(self, t=None,quantity=1):
        if t is not None:
            self.update(t)
        return np.random.normal(self.mean,self.std,(quantity,self.dimension))
    
    def update(self,t):
        if self.updator is not None:
            self.mean, self.std = self.updator(t)

    @property
    def time_dependant(self):
        return self.updator is not None

class Discrete1DUniformGenerator(Generator):
    def __init__(self,n):
        super().__init__()
        self.n = n
    
    def __call__(self,t=None,quantity=1):
        return np.random.randint(0,self.n,quantity)

    def update(self,t):
        pass

    @property
    def time_dependant(self):
        return False
    
class UniformGenerator(Generator):
    def __init__(self,a,b,updator=None,t=0):
        super().__init__()
        self.a = a 
        self.b = b
        self.updator = updator
        # initialize with updator at time=t
        self.update(t)
    
    def __call__(self, t=None, quantity=1):
        if t is not None:
            self.update(t)
        return (self.a-self.b)*np.random.random((quantity,len(self.a)))+self.b

    def update(self,t):
        if self.updator is not None:
            a, b = self.updator(t)

    @property
    def time_dependant(self):
        return self.updator is not None

class CustomGenerator(Generator):
    def __init__(self, f, time_dependant=False):
        super().__init__()
        self.f = f
        self.time_dependant = time_dependant

    def __call__(self,t=None,quantity=1):
        if self.time_dependant:
            return self.f(t=t,quantity=quantity)
        else:
            return self.f(quantity=quantity)

    def update(self,t):
        pass

class SumGenerator(Generator):
    def __init__(self,generators):
        super().__init__()
        self.generators = generators

    def __call__(self,t=None,quantity=1):
        return np.sum(np.array([generator(t,quantity) for generator in self.generators]),axis=0)

    def update(self,t):
        for generator in self.generators:
            generator.update(t)
    
    @property
    def time_dependant(self):
        result=False
        for gen in self.generators:
            result = result or gen.time_dependant
        return result


class UnionGenerator(Generator):
    def __init__(self,generators,dim=None,probas=None):
        if probas is None:
            probas = np.ones(len(generators))/len(generators)
        assert abs(sum(probas)-1)<1e-3 #assert that the sum of probas is 1
        self.probas = probas
        self.generators = generators
        if dim is not None:
            self.dim = dim
        else:
            self.dim = len(self.generators[0]()[0])

    def __call__(self,t=None,quantity=1):
        u = np.random.random(quantity)
        X = np.zeros((quantity,self.dim),float)
        acc=0
        for idx,p in enumerate(self.probas):
            indices = np.nonzero((acc<=u)*(u<acc+p+(idx==len(self.generators)-1)))
            X[indices,:] = self.generators[idx](quantity=len(indices[0]))
            acc+=p
        return X
    
    def update(self,t):
        for generator in self.generators:
            generator.update(t)
    
    @property
    def time_dependant(self):
        result=False
        for gen in self.generators:
            result = result or gen.time_dependant
        return result



class DataGenerators:
    def __init__(self,generators,class_generator=None):
        self.generators = generators
        if class_generator is None:
            self.class_generator = Discrete1DUniformGenerator(len(generators))
        else:
            self.class_generator = class_generator
        self.dimension = len(self.generators[0]()[0])

    @property
    def nb_class(self):
        return len(self.generators)

    def generate_data(self,quantity, t=None):
        if t is not None:
            self.update(t)
        X = np.zeros((quantity,self.dimension),float)
        y = self.class_generator(quantity=quantity)
        for cl in range(self.nb_class):
            cl_indices = np.nonzero(y==cl)
            X[cl_indices,:] = self.generators[cl](quantity=len(cl_indices[0]))
        return X, y

    def update(self,t):
        for generator in self.generators:
            generator.update(t)

    @property
    def time_dependant(self):
        result=False
        for gen in self.generators:
            result = result or gen.time_dependant
        return result


class DataGeneratorsWithHiddenFunction:
    def __init__(self,generator,hidden_f,time_dependant_function=False,nb_class=2):
        self.generator = generator
        self.hidden_f = hidden_f
        self.time_dependant_function = time_dependant_function
        self.nb_class = nb_class
        self.dimension = len(self.generator())

    def generate_data(self,quantity, t=None):
        if t is not None:
            self.generator.update(t)
        X = self.generator(quantity=quantity)
        if self.time_dependant_function:
            y = self.hidden_f(X,t)
        else:
            y = self.hidden_f(X)
        return X, y
    
    @property
    def time_dependant(self):
        return self.time_dependant_function or self.generator.time_dependant
    

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

def unit_sphere_test(dimension,quantity,noise=0.1):
    x = np.random.normal(0,1,(quantity,dimension))
    x = np.random.normal(1,noise,quantity)[:,None]*x/(np.linalg.norm(x,axis=1)+1e-6)[:,None]
    return x


def test():
    generators = []
    generators.append(GaussianGenerator(np.zeros(2),0.3))
    generators.append(UniformGenerator(np.array([-3,-3]),np.array([-1,-1])))
    f = lambda quantity:unit_sphere_test(2,quantity)
    generators.append(SumGenerator([CustomGenerator(f),GaussianGenerator(np.zeros(2),0.1)]))
    generators.append(UnionGenerator([GaussianGenerator(np.array([2,-2]),0.3),GaussianGenerator(np.array([3,-3]),0.2)],[0.7,0.3]))
    class_generator = Discrete1DUniformGenerator(len(generators))
    return DataGenerators(generators,class_generator)

def test2():
    generator = UniformGenerator(np.zeros(2),np.ones(2))
    f = lambda X: X[:,0]+X[:,1]>1
    return DataGeneratorsWithHiddenFunction(generator,f)

def separation_on_uniform(dim=2):
    generator = UniformGenerator(np.zeros(dim),np.ones(dim))
    f = lambda X: np.sum(X,axis=-1)>X.shape[-1]/2
    return DataGeneratorsWithHiddenFunction(generator,f)

def xor_gen(dim=2,std=0.5):
    generators = []
    mean = np.zeros(dim)
    mean[0] = 1
    mean[1] = 1
    generators.append(UnionGenerator([GaussianGenerator(mean.copy(),std,dim),GaussianGenerator(-mean.copy(),std,dim)]))
    mean[0] = -1
    generators.append(UnionGenerator([GaussianGenerator(mean.copy(),std,dim),GaussianGenerator(-mean.copy(),std,dim)]))
    return DataGenerators(generators)


def two_gaussians(dim,std):
    target = np.zeros(dim)
    target[0] = 1
    lst_gen = []
    lst_gen.append(GaussianGenerator(target,std))
    lst_gen.append(GaussianGenerator(-target,std))
    return DataGenerators(lst_gen)

def four_gaussians(dim=2,std=0.5):
    target = np.zeros(dim)
    target[0] = 1
    lst_gen = []
    lst_gen.append(GaussianGenerator(target.copy(),std))
    lst_gen.append(GaussianGenerator(-target.copy(),std))
    target[0] = 0
    target[1] = 1
    lst_gen.append(GaussianGenerator(target.copy(),std))
    lst_gen.append(GaussianGenerator(-target.copy(),std))
    return DataGenerators(lst_gen)

def not_convex(dim=2,noise=0.1):
    generators = []
    center_std = 0.7/np.sqrt(dim)
    generators.append(GaussianGenerator(np.zeros(dim),center_std))
    f = lambda quantity:unit_sphere_test(dim,quantity,noise)
    generators.append(CustomGenerator(f))
    return DataGenerators(generators)

def display_classes(X,y):
    color_names = "rgbcmykw"
    colors = [color_names[i] for i in y]

    plt.scatter(X[:,0], X[:,1], color = colors)
    plt.show()



if __name__ == "__main__":

    data_generator = four_gaussians(2,0.5)

    X,y = data_generator.generate_data(1000)

    display_classes(X,y)

    


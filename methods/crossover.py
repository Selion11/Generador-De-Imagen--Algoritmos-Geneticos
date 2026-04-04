import numpy as np
from abc import ABC, abstractmethod

class CrossoverMethod(ABC):
    @abstractmethod
    def crossover(self, p1_genes, p2_genes):
        """
        Cruza dos padres y retorna los genes hijas.
        :param p1_genes: Genes del padre 1 (numpy array)
        :param p2_genes: Genes del padre 2 (numpy array)
        :return: Tupla con genes de los hijos (child1_genes, child2_genes)
        """
        pass

class OnePointCrossover(CrossoverMethod):
    def crossover(self, p1_genes, p2_genes):
        cut = len(p1_genes) // 2
        
        c1_genes = np.concatenate([p1_genes[:cut], p2_genes[cut:]])
        c2_genes = np.concatenate([p2_genes[:cut], p1_genes[cut:]])
        
        return c1_genes, c2_genes
    
class TwoPointCrossover(CrossoverMethod):
    def crossover(self, p1_genes, p2_genes):
        length = len(p1_genes)
        
        if length < 2:
            return p1_genes.copy(), p2_genes.copy()

        puntos = np.random.choice(range(1, length), size=2, replace=False)
        cut1, cut2 = sorted(puntos) 

        c1_genes = np.concatenate([
            p1_genes[:cut1],
            p2_genes[cut1:cut2],
            p1_genes[cut2:]
        ])
        
        c2_genes = np.concatenate([
            p2_genes[:cut1],
            p1_genes[cut1:cut2],
            p2_genes[cut2:]
        ])
        
        return c1_genes, c2_genes
    
class UniformCrossover(CrossoverMethod):
    def __init__(self, probability=0.5):
        """
        :param probability: Probabilidad (p) de que el Hijo 1 herede el gen del Padre 1.
                            Por defecto es 0.5 (50% equitativo).
        """
        self.p = probability

    def crossover(self, p1_genes, p2_genes):
        mask = np.random.rand(len(p1_genes)) < self.p
        
        c1_genes = np.empty_like(p1_genes)
        c2_genes = np.empty_like(p2_genes)
        
        c1_genes[mask] = p1_genes[mask]
        c1_genes[~mask] = p2_genes[~mask]
        
        c2_genes[mask] = p2_genes[mask]
        c2_genes[~mask] = p1_genes[~mask]
        
        return c1_genes, c2_genes
    
class AnnularCrossover(CrossoverMethod):
    def crossover(self, p1_genes, p2_genes):
        length = len(p1_genes)
        
        if length < 2:
            return p1_genes.copy(), p2_genes.copy()

        start_idx = np.random.randint(0, length)
        
        L = np.random.randint(1, max(2, length // 2 + 1))
        
        c1_genes = p1_genes.copy()
        c2_genes = p2_genes.copy()
        
        for i in range(L):
            idx = (start_idx + i) % length 
            
            c1_genes[idx] = p2_genes[idx]
            c2_genes[idx] = p1_genes[idx]
            
        return c1_genes, c2_genes

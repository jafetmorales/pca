#THIS IS AN EXAMPLE OF HOW TO USE THE PCAFEATUREDUCTION FUNCTION
#WRITTEN BY JAFET MORALES

from numpy import array
from pcaLibrary import pcaFeatureReduction

#each row is a feature or variable
#each column is a sample of the data or observation
dataAll=array([[1,2,3],[-5,5,6],[7,70,9],[99,-5,33]])
print dataAll

dataReduced=pcaFeatureReduction(dataAll)
print dataReduced

from assignment import Assignment

assignment_2 = Assignment("./data/tp2_data.csv")

ks = [7, 51, 96]
num_components = [7, 51, 96]
epsilons = [23.337, 36.726, 69.565]

# assignment_2.k_means(ks)    #Kmeans possible aplication -> Vector quantization
# assignment_2.gaussian_mix(num_components) #GMM possible application -> Generating new points based on the distribution
assignment_2.dbscan(epsilons = epsilons)  #DBSCAN possible application -> Identifying "denser" regions in terms of seisms

from assignment import Assignment

assignment_2 = Assignment("./data/tp2_data.csv")

ks = [10, 20, 30]   #TODO por aqui valores que façam sentido para vector quantitization (nao sei como se escreve xD)
num_components = [2, 4, 6]  #TODO por aqui valores que façam sentido (provavelmente pode ser a mesma justificação que o kmeans)

# assignment_2.k_means(ks)
# assignment_2.gaussian_mix(num_components)

epsilons = [23.337, 36.726, 69.565] #TODO adicionar mais epsilons
assignment_2.dbscan(epsilons = epsilons)
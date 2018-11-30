from assignment import Assignment

assignment_2 = Assignment("./data/tp2_data.csv")

ks = [10, 20, 30]
num_components = [2, 4, 6]

# assignment_2.k_means(ks)
assignment_2.gaussian_mix(num_components)
# assignment_2.dbscan()
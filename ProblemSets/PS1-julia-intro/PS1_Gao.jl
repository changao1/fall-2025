# generate random matrix with 500 rows and 50 columns
A = rand(500, 50)
# compute the mean of each column
col_means = mean(A, dims=1)
# center the matrix by subtracting the column means
A_centered = A .- col_means
# compute the covariance matrix
cov_matrix = cov(A_centered, dims=1)
# compute the eigenvalues and eigenvectors of the covariance matrix
eigen_decomp = eigen(cov_matrix)
# extract the top 5 eigenvalues and corresponding eigenvectors
top_indices = sortperm(eigen_decomp.values, rev=true)[1:5]
top_eigenvalues = eigen_decomp.values[top_indices]
top_eigenvectors = eigen_decomp.vectors[:, top_indices]
# project the centered data onto the top 5 eigenvectors
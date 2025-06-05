# implement two regularizations:
# 1. (orthogonality) the rowspaces of the Jacobians of the encoder with respect to its parameters 
# at different observations should be orthogonal to each other
# 2. (full row rank) the Jacobians should have full row rank
# mathematically, R_orth = ||J_iJ_j^T||^2 where J_i is the Jacobian at observation i
# J_j is the Jacobian at observation j != i
# R_full = -log det(J_iJ_i^T + epsilon I) It should not be active when the determinant is already large

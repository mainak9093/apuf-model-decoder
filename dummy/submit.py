import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
	
	return G


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors
	
	return a, b, c, d, p, q, r, s


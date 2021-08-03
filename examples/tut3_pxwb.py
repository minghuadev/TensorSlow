#!/usr/bin/env python3

# code from the jupyter_notebook

from tensorslow.graph import Graph, placeholder, Variable
from tensorslow.operations import add, matmul, sigmoid
from tensorslow.session import Session

# Create a new graph
Graph().as_default()

# Create variables
# Create placeholder
# Create hidden node y
# Create output node z
x = placeholder()
w = Variable([1, 1])
b = Variable(0)
p = sigmoid( add(matmul(w, x), b) )

session = Session()
output = session.run(p, {
    x: [3, 2]
})

print("output: ", output)


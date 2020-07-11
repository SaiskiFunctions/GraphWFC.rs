# Propagation

`NodeValue = i32`
`EdgeDirection = i32`
`node_values: Set<NodeValue>`: Set of possible values that a node could take
`output_matrices: Map<EdgeDirection, AdjencyMatrix>`:
  `diagonal`: Stores collapsed node output values
  `non-diagonal`: Stores relationships between nodes
`rules: Map<Tuple<EdgeDirection, NodeValue>, Set<NodeValue>>`: Stores which nodes can be next to other nodes in a direction

## Example Process (Images)

INPUT: Image, Output Size

Preamble (Transform Image into Graph):
Image -> Graph
Graph -> Rules
Collect collapse calculation metrics
        classic wfc: node frequency
Set up an output structure of adjency matrices

The Meat (Wave Function Collapse):
loop:
  if can propagate:
    propagate
  else if can collapse:
    collapse
  else
    output

Renderer (Transform output into Image):
Ouput array -> Image


## Propagation

|  | 1 | 2| 3|
|1 |   |  | | |
| 2|   |   | | |
|3|

1. Pick node to collapse with lowest entropy
2. Propagate collapse






Adjency matrix structures the relationships, its only used for propagating information

Adjency matrices have to have a subset (or equal) relationships to the input.

Turning input into
Learning rules from input graph
Building an output structure of adjency matrices
Applying propagation

Unions and intersections 
possibilities are output as a set
set logic stuff tm

Collapse calculation is arbitrary <- 

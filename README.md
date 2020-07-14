# Propagation

`vertexLabel = i32`
`EdgeDirection = i32`
`vertex_labels: Set<vertexLabel>`: Set of possible labels that a vertex could take
`output_matrices: Map<EdgeDirection, AdjencyMatrix>`:
  `diagonal`: Stores collapsed vertex output labels
  `non-diagonal`: Stores relationships between vertices
`rules: Map<Tuple<EdgeDirection, vertexLabel>, Set<vertexLabel>>`: Stores which vertices can be next to other vertices in a direction

## Example Process (Images)

INPUT: Image, Output Size

Preamble (Transform Image into Graph):
Image -> Graph
Graph -> Rules
Collect collapse calculation metrics
        classic wfc: vertex frequency
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


```
==== RUN COLLAPSE FUNCTION ====
INPUT: input_graph, output_graph, retry_count

rules = input_graph.rules()
try_count = 0
WHILE try_count <= retry_count:
    MATCH COLLAPSE_ALGORITHM(rules, output_graph.clone()) {
        Some(result) => result
        None => try_count += 1; continue
    }

==== COLLAPSE ALGORITHM ====
INPUT: input_graph_rules, output_graph
RETURN Option<Graph>:
    Some: output_graph (collapsed)
    None:
HEAP: BinaryHeap<OBSERVE_ACTION>
GEN_OBSERVE: HashSet<VertexIndex>
OBSERVED: HashSet<VertexIndex>
PROPAGATIONS: Vec<PROPAGATE_ACTION>
1. Create a new binary HEAP.
2. Create a number of OBSERVE_ACTION's equal to the number of vertices onto the HEAP.
3. IF length of OBSERVED == Graph.vertices.length:
    return SUCCESS!
4. IF HEAP is empty:
    1. return SUCCESS!
   ELSE IF PROPAGATIONS is empty:
    2. IF GEN_OBSERVE is not empty:
        1. Add OBSERVE_ACTION's onto the HEAP FOR EACH vertexIndex in GEN_OBSERVE and empty GEN_OBSERVE set.
    3. Pop an OBSERVE_ACTION off of the HEAP
    4. Collapse the vertexLabel of the vertexIndex this action references to a singleton set.
    5. IF the vertexLabel at vertexIndex changed:
        2. Find vertices connected to this vertexIndex using the out_graph edges property.
        3. For each connected vertex, push a PROPAGATE_ACTION to PROPAGATIONS.
    6. GOTO 3.
   ELSE:
    7. Pop a PROPAGATE_ACTION off of PROPAGATIONS.
    8. Constrain vertexLabel according to propagation.
    9. IF the vertexLabel at vertexIndex changed:
        1. IF vertexLabel == {}:
            1. return FAILURE!
           ELSE IF vertexLabel != singleton:
            2. Add vertexIndex to GEN_OBSERVE.
           ELSE:
            3. Add vertexIndex to OBSERVED.
        2. Find vertices connected to this.vertexIndex using the out_graph edges property.
        3. For each connected vertex, push a PROPAGATE_ACTION to PROPAGATIONS.
    10. GOTO 3.


=== EXAMPLE ===

1abc --- 2abc       HEAP: collapse(0), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: []
|        |          OBSERVED: []
0abc --- 3abc

RUN collapse(0) {

}

1abc --- 2abc       HEAP: propagate(0, 1), propagate(0, 3), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: []
|        |          OBSERVED: [0]
0a ----- 3abc

RUN propagate(0, 1) {

}

1ab ---- 2abc       HEAP: propagate(1, 2), propagate(0, 3), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: [1]
|        |          OBSERVED: [0]
0a ----- 3abc

RUN propagate(1, 2) {

}

1ab ---- 2c         HEAP: propagate(2, 3), propagate(2, 1), propagate(0, 3), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: [1]
|        |          OBSERVED: [0, 2]
0a ----- 3abc

RUN propagate(2, 3) {
    
}

1ab ---- 2c         HEAP: propagate(2, 1), propagate(0, 3), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: [1, 3]
|        |          OBSERVED: [0, 2]
0a ----- 3ab

RUN propagate(2, 1) {
    
}

1ab ---- 2c         HEAP: propagate(0, 3), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: [1, 3]
|        |          OBSERVED: [0, 2]
0a ----- 3ab

RUN propagate(0, 3) {
    
}

1ab ---- 2c         HEAP: collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: [1, 3]
|        |          OBSERVED: [0, 2, 3]
0a ----- 3b

PEAK collapse {
    RUN GEN_OBSERVE
}

1ab ---- 2c         HEAP: collapse(1), collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: []
|        |          OBSERVED: [0, 2, 3]
0a ----- 3b

RUN collapse(1) {

}

1b ----- 2c         HEAP: collapse(1), collapse(2), collapse(3)
|        |          GEN_OBSERVE: []
|        |          OBSERVED: [0, 1, 2, 3]
0a ----- 3b

ALL vertices OBSERVED -> RETURN
```

```
Why are we using this data structure for connections and directions?

// (0: N, 0: a) -> (1: b)
// (0: N, 1: b) -> (2: c)
// (1: S, 1: b) -> (0: a)
// (1: S, 2: c) -> (1: b)
// (2: E, 0: a) -> (1: b)
// (2: E, 1: b) -> (2: c)
// (3: W, 1: b) -> (0: a)
// (3: W, 2: c) -> (1: b)

This means when we look up further propagations from a collapse or a propagate action we have to log
the label that has been propagated/collapsed, then go to the graph's edges, iterate through it and look
in each label in the edges map for hash_sets that contain the desired "from" vertex as their first label
and do this for four separate entries (more if we are in higher dimensions) in the map.

Could we not just combine all of this edge and connection information into a single (sparse) matrix making it
much simpler to look up connections and generate rules.

    1b --- 2c
    |      |
    0a --- 3b

    1:n, 2:s, 3:e, 4:w, 5, 6, 

    0     1     2     3   <-- FROM
 ━╋━━━━━╋━━━━━╋━━━━━╋━━━━
0 ┃ 0   ┃ 2:S ┃ 0   ┃ 4:W
 ━╋━━━━━╋━━━━━╋━━━━━╋━━━━
1 ┃ 1:N ┃ 0   ┃ 4:W ┃ 0
 ━╋━━━━━╋━━━━━╋━━━━━╋━━━━
2 ┃ 0   ┃ 3:E ┃ 0   ┃ 1:N
 ━╋━━━━━╋━━━━━╋━━━━━╋━━━━
3 ┃ 3:E ┃ 0   ┃ 2:S ┃ 0
⟰
TO

Then store as sparse matrix and query directly to get vertex connection AND direction information immediately.
```




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

Modularising graph metadata generation:
Method on graph:
pub fn metadata(&self, function<Graph, U>) -> U 

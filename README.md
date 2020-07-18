[![travis build](https://travis-ci.org/dpwdec/wfc-rust.svg?branch=master)](https://travis-ci.org/github/dpwdec/wfc-rust)

# Wave Function Collapse



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
all_labels = input_graph.all_labels()
try_count = 0
rng.seed.start
WHILE try_count <= retry_count:
    MATCH COLLAPSE_ALGORITHM(rng, rules, frequencies, all_labels, output_graph.clone()) {
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
2. FOR EACH vertex in output_graph: // Create a number of OBSERVE_ACTION's equal to the number of vertices onto the HEAP.
    1. IF vertex labels is a <proper subset> of all_labels:
        1. Find vertices connected to this.vertexIndex using the out_graph edges property.
        2. For each connected vertex, push a PROPAGATE_ACTION to PROPAGATIONS.
        3. IF vertex labels is a singleton set:
            1. Add vertexIndex to OBSERVED
            2. CONTINUE
    2. Push vertex OBSERVE_ACTION onto the HEAP.
3. IF length of OBSERVED == Graph.vertices.length:
    return SUCCESS!
4. IF HEAP is empty:
    1. return SUCCESS!
   ELSE IF PROPAGATIONS is empty:
    2. IF GEN_OBSERVE is not empty: // drain
        1. Add OBSERVE_ACTION's onto the HEAP FOR EACH vertexIndex in GEN_OBSERVE and empty GEN_OBSERVE set.
    3. Pop an OBSERVE_ACTION off of the HEAP
    4. IF OBSERVE_ACTION's vertexIndex is NOT in OBSERVED:
        1. Collapse the vertexLabel of the vertexIndex this action references to a singleton set.
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
        2. For each connected vertex.
            1. If vertex is not in OBSERVED:
                1. push a PROPAGATE_ACTION to PROPAGATIONS
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

![Shannon entropy](entropy_equation.svg "{\displaystyle \mathrm {H} (X)=-\sum _{i=1}^{n}{\mathrm {P} (x_{i})\log _{b}\mathrm {P} (x_{i})}}")

H(X) -> shanon entropy

Sum of expression

P is probability for a element in a set of possible values

Map {
    0: 4,
    1: 5,
    2: 10,
    3: 20
}

{0, 1} P(0) = 4/4+5

{}

bc
ab

b -> c -> a -> b

=>

1-2
| |
0-1

=>

input conforms to a linear space for input

🍂🍂🍂🍂🍂🍂🍂
🍂🍂🍂🍂🍂🍂🍂
🌳🌳🌳⛩🌳🌳🌳
🌊🌊🌊🌊🌊🌊🌊
🌊🌊🌊🌊🌊🌊🌊
🌊🌊🌊🌊🌊🌊🌊

😅😅😅😅😅😅😅
😅😅😅😅😅😅😅
🌳🌳🌳⛩🌳🌳🌳
🌊🌊🌊🌊🌊🌊🌊
🌊🌊🌊🌊🌊🌊🌊
🌊🌊🌊🌊🌊🌊🌊

# Parser / Renderer

Pass `Parser-Renderer` pair to the algorithm constructor. But both parts can be switched out.

You have to provide a mapping between input and output labels.


The input and output must be fully mapped. The output space can be "smaller" than the input space. Meaning we can map multiple input elements to a single output element.

injective function - mapping from set A to set B, every element of A goes to an element B, its a one way mapping
 
Input labels -> Output labels (Content Mapping)
Input directions -> Output Directions (Structural Mapping)

Mapping can be applied before or after the actual collapse algorithm.

PMR

pixel data -> graph (parser)
mapping: optional
collapse
graph -> pixel data (renderer)

quad directional pixel space NSEW -> 
bi directional pixel space LR: NS -> L EW -> R

renderer

homomorphism 

1. Parse input to input graph
2. Optionally map input graph to output formatted input graph
3. Run collapse on output graph with input data
4. Render the output into the output format

    Pixels(Red, Black) -> 
    Graph(Red -> 0, Black -> 1)
    Graph(0 -> 1, 1 -> 1) (Black: 1) -> |-------+
    Derive Rules, Frequencies, All labels ->    |
    Collapse Output graph ->                    |
    Render Output graph ->  <-------------------+
    Pixels(Black)

No mapping provided: renderer is INVERSE of parser

@ -> 0
& -> 1

@ -> &
& -> &

fn pix_to_graph(pixel_data) -> graph, output_key

output_key

collapse stuff here


fn graph_to_pix(graph, output_key) -> pixel_data


@@
&&

@ -> &
fn text_to_graph(string) -> graph, key {
    key: HashMap;

    string.
}

Parser-Mapping-Renderer 4x4






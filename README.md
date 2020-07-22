[![Build Status](https://travis-ci.org/dpwdec/wfc-rust.svg?branch=master)](https://travis-ci.org/dpwdec/wfc-rust)


A graph based implementation of the Wave Function Collapse algorithm. This project improves on the flexibility and speed of the algorithm by decoupling its constraint solving functionality from input / output data and generalising parsing and rendering into graph structures allowing for speedy constraint propagation and easy conversion between media types.

This implementation:
1. Generalises and brings clarity to the constraint solving core of the algorithm.
2. Allows for input / output between arbitrary media types. For example, inputting image data for constraint generation and outputting sound data.
3. Modularises the structure of algorithm to improve run time on distributed services and make profiling and performance optimisation easier.

## Background

The Wave Function Collapse algorithm is a constraint solving algorithm created by Maxim Gumin based on Paul Merrell's work in model generation and Paul F. Harrison's work in texture synthesis. Its primary application has been in generative media and games as a method for procedurally generating large amounts of original content from a small set of human defined inputs.

`INPUT OUTPUT IMAGE EXAMPLE HERE`

Since its release the algorithm has been ported across to many languages and content creation systems, however, because of the algorithms traditional area of application it has generally been implemented with fairly obfuscated data structures (such as nested three dimensional arrays) and strongly coupled with the process of parsing in input data making it difficult for users to understand how the constraint solving elements of the algorithm work and how to implement the algorithm in arbitrary n dimensional spaces and across many media types.

## Method

This implementation of WFC uses only Graph structures in the algorithm's `collapse` loop where the core constraint solving functionality occurs.

Graphs are an incredibly flexible data (and mathematical) structure that can be used to represent a wide range of input structures. We can use the edge-vertex structure of Graphs to represent n-dimensional spaces through n number of directed edges. For example, in a 1-dimensional space (a line), graph vertexes are connected by edges labelled with forwards and backwards. This approach generalises to higher input dimensions with the number of edge labels correlating with the number of discrete directions across each dimension. Furthermore, Graph structures can encode non-spatial discrete relationships, such as sequential relationship between words and word classes, or even temporal changes.

It follows that anything that can be reasonably parsed into Graph data can be constraint solved with this single implementation of the WFC algorithm; images, music, text, topological data etc. all with a single `collapse` loop. There are several advantages to this:
1. Code is DRY-er as a result and more easily maintainable.
2. Domain specific implementation details are entirely decoupled from the generative elements of the algorithm making it far more flexible.
3. The "hottest" part of the code is small and modularised making profiling and optimisation much easier.
4. Extending and debugging the algorithms core functionality is centralised into a single set of modules. Extending Graph `collapse` functionality once extends functionality for all possible input types.

The algorithm is also generalised to be agnostic as to the state of the graph that is submitted for collapse, with the constraint solving core able to accept partially collapsed graphs as input as well as entirely uncollapsed graphs. This implementation of a more general input schema allows the aglorithm to:
1. Support checkpointing and spot processes in the code. The `collapse` process is a fairly computationally intensive task, so, should a process fail or have to be killed at some point along its run time the constraint `collapse` process will accept a partially collapsed Graph as input and continue constraint solving from that point
2. Work effectively with manipulated Graph data between the parse and collapse stages. Should you want to run further manipulations on Graph data, such as combining or filtering graphs, that alter the structure of the Graph or labels then `collapse` will still accept a manipulated graph as input to solve.

A final piece of modularisation is the entropy function and observation sorting. We implemented Shanon Entropy as our complexity heuristic, however the aglorithm's complexity heuristic is entirely modular. Sorting the complexity of vertices is done using a BinaryHeap data structure (as a min heap) such that the lowest entropy vertex to be observed is always at the top of the heap. This means, that we can easily implement and extend different methods for defining a complexity heuristic for vertices in the Graph as sorting is done automatically by the BinaryHeap.

# Structure

This section describes how each different part of the algorithm works. A primary goal with this implementation was to make WFC more transparent and extensible for other developers, if you are confused about the terms or concepts used here please see the `Q&A` section below where I attempt to explain some of the foundational concepts. If you are not at all familiar with WFC it might be helpful to start there.

## Overview

The algorithm takes in some form of input (image, sound, model data etc.) parses that input into a graph structure (the `input graph`) and then uses that graph structure to derive a set of constraints. Next the algorithm creates a differently sized (usually much larger) graph structure (the `output graph`) that has its vertices set to a superposition of all possible labels in the input graph. 

It then proceeds to collapse each set of labels at a vertex down to a single label by computing the entropy at a vertex's position and collapsing the least entropic vertices first. Finally when all vertices have only a single label the algorithm parses the `output graph` back into some useful form of real world output (not necessarily the same as input media type) and outputs it to the user.

```
┍━━━━━━━━━━━━━┑
│ Input Media │
┕━━━━━━━━━━━━━┙
       ⬇ (Parsed to)
┍━━━━━━━━━━━━━┑
│ Input Graph │
┕━━━━━━━━━━━━━┙
       ⬇
┍━━━━━━━━━━━━━━━━━━━━━━━━━┑
│ Derive Constraint Rules │
┕━━━━━━━━━━━━━━━━━━━━━━━━━┙
       ⬇
┍━━━━━━━━━━━━━━━━━━━━━┑
│ Create Output Graph │
┕━━━━━━━━━━━━━━━━━━━━━┙
       ⬇
┍━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│ Collapse Graph (Core WFC)│
┕━━━━━━━━━━━━━━━━━━━━━━━━━━┙
       ⬇ (Rendered to)
┍━━━━━━━━━━━━━━┑
│ Output Media │
┕━━━━━━━━━━━━━━┙
```

## Parser-Renderer Pairs

The parser-renderer pair function as a prelude and coda to the core `collapse` algorithm making rule specification and i/o easier for human users. When input is submitted by a user it must come with a parser-renderer pair and an optional mapping.

- `Parser`: implements a process that accepts some input and outputs a Graph and Map that describes how to decode Graph.
- `Renderer`: implements a process that takes a Graph and a Map as input and uses it render the Graph into a human readable output.
- `Mapping`: An optional secondary Map that maps between differnent media types of dimensional spaces.

## The `collapse` Constraint Solving Loop

### Overview

At its core this implementation of WFC takes some uncollapsed graph, assigns an entropy value to each vertex on the graph based on its set of labels and uses a process of elimination to remove possibilities of vertex positions until the algorithm reaches a contradition, in which case it fails or the graph is fully collapsed.


```
LOOP:
  IF the information in the graph changed due to a collapse or propagation:
    PROPAGATE information (constraints) between connected vertices

  ELSE IF there are vertices to collapse
    COLLAPSE the vertex with the lowest entropy

  ELSE IF there are no vertices left to collapse
    OUTPUT the collapsed graph

```
In the interest of simplicty and to help readers to understand the core solving loop of the algorithm the pseudo code above does not show the failure case which occures if a propagation constrains the set of possible labels at a vertex to be empty (i.e. a labels set of size `0`) which would mean that the algorithm reached a contradiction in its constraint solving and failed. This check for a label set of `0` occurs after the `PROPAGATE` step.

### Data Structures

- `Observe`: Represents a possible observation and the entropy of a vertex to be observed at the time it was created.
- `Propagate`: Represents a propagation `from` a vertex `to` another vertex in along a a directed edge.
- `Heap: BinaryHeap<Observe>`: A [min heap] of `Observe` actions storing the Observe with the lowest entropy on top.
- `Observed: Set<VertexIndex>`: A list recording which vertices have been collapsed to a singleton set.
- `Propagations: Vec<Propagate>`: A stack of `Propagate` actions representing a queue of constraint propagations that need to take place.
- `gen_observe: Vec<VertexIndex>`: An array of vertexes the entropy of which has changed during the propagation process and need to have new `Observe` operations generated for them.

### Initialization
This the pseudocode for initialization of the algorithm. The main function of this code is to initialise the `Heap` of `Observe` actions and the initial entropy of the Graph's vertices as well as allow for partially collapsed Graphs as input.
```
FOR EACH vertex IN output_graph:
    IF vertex.labels EQUAL TO PROPER SUBSET OF all_labels:
        FOR EACH other_vertex IN vertex.connections:
            PUSH Propagate.new ONTO Propagations

    IF vertex.labels IS SINGLETON SET:
        ADD vertex TO Observed
        CONTINUE // Skips last command in this FOR EACH
    
    PUSH Observe.new FOR vertex ONTO Heap
```

### Constraint Solving `collapse` Loop Deep Dive
This pseudocode shows the basic logic for the core loop of the `collapse` algorithm and how constraint solving is used to collapse the Graph. 
```
LOOP:
    IF Observed.length == Graph.vertices.length:
        RETURN Graph // Success!

    IF Heap.is_empty:
        RETURN Graph // Success!

    ELSE IF Propagations.is_empty:
        IF gen_observe.not_empty:
            FOR EACH vertex IN gen_observe:
                PUSH Observe.new FOR vertex ONTO Heap
            
            EMPTY gen_observe

        POP Observe OFF Heap // Returns lowest entropy Observe

        COLLAPSE Observe.vertex.labels // Labels -> Singleton Set
        FOR EACH other_vertex IN Observe.vertex.connections:
            PUSH Propagate.new ONTO Propagations
            
    ELSE:
        POP a Propagate OFF Propagations

        COMPOSE SET OF LABEL constraints // Set up constraints

        INTERSECT Propagate.to_vertex.labels WITH constraints
        IF Propagate.to_vertex.labels CHANGED:
            IF Propagate.to_vertex.labels.is_empty:
                RETURN Error // Failure!
            ELSE IF Propagate.to_vertex.labels.is_singleton:
                ADD Propagate.to_vertex TO observed
            ELSE:
                ADD Propagate.to_vertex TO gen_observe
            
            FOR EACH other_vertex IN Observe.to_vertex.connections:
                IF other_vertex NOT IN Observed:
                    PUSH Propagate.new ONTO Propagations
```

## Method

#### How does using graphs improve the algorithm?

Because Graphs are inherently n-dimensional it means that implementing the algorithm for any n-dimensional graph removes the need to re-write the algorithm for different input media types. Each new media type only needs to have a renderer parser pair added to it

**What is a graph?**

The term graph used in the context of this implementation of the algorithm refers to the definition of a graph used in the mathematical field of Graph Theory. A graph is a structure consisting of *vertices*, *edges* and *labels*. The set of vertices of a graph are connected by a set edges. Each vertex can be *labelled* with a value. Edges can also be labelled with a direction.

In the `simple graph` example below there are six vertices connected by a number of edges. The vertices are labelled with numbers 1 through 6.

`A simple graph`:
<a href="https://en.wikipedia.org/wiki/Graph_theory#/media/File:6n-graf.svg">
<img src="basic_graph.png" alt="Graph" style="height:150px;">
</a>

It's important to understand the label of a vertex is separate from absolute information about a vertex. The `text rendered` graph below makes this more clear. There are four vertices in this graph indexed 1 through 4, however the *labels* of these vertices are `a`, `b`, and `c`. I.e. the label of the vertex identified as `1` is `a`. 

`A graph rendered as text`:
```
1a --- 2b
|      |
|      |
3c --- 4a
```

**What does the core constraint algorithm actually do?**

The algorithm takes in an uncollapsed graph and outputs a collapsed graph based on a set of constraint solving rules.

**What is an uncollapsed graph?**

An uncollapsed grap

#### What is entropy and how is it calculated?

The algorithm takes in some form of input (image, sound, model data etc.) parses it into a graph structure, deri
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

## Pipeline

This implementation follows a process of


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
```

```
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






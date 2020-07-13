mod utils;
use utils::{hash_map, hash_set};
mod graph;
use graph::*;

use std::collections::{HashSet, HashMap, BinaryHeap};
use std::iter::FromIterator;
use std::hash::Hash;
use std::cmp::{Eq, Ordering};
use std::clone::Clone;

/*
Each node in a graph has an index,
There is a directional relationship between nodes
Each node has a value
*/

fn main() {

    // Preamble
    /*
    1b --- 2c
    |      |
    0a --- 3b

    North = 0, South = 1, East = 2, West = 3

    a = 0, b = 1, c = 2
    */

    // turn input into a graph

    // index number represents each node position

    // generate the rules
    // Set up uncollapsed output values

    // generate output graph from input graph

    // let output_graph_nodes = vec![hash_set(&[0, 1, 2]); 4];
    //
    // let output_graph_edges: HashMap<NodeIndex, Vec<(NodeIndex, EdgeDirection)>> = hash_map(&[
    //     (0, vec![(1, 0), (3, 2)]),
    //     (1, vec![(0, 1), (2, 2)]),
    //     (2, vec![(3, 1), (1, 3)]),
    //     (3, vec![(0, 3), (2, 0)])
    // ]);
    //
    // let output_graph = Graph::new(
    //     output_graph_nodes,
    //     output_graph_edges
    // );

    /*
    1)

    1abc --- 2abc       HEAP: collapse(0), collapse(1), collapse(2), collapse(3)
    |        |
    |        |
    0abc --- 3abc

    Call: HEAP --> collapse(0) PUSH TO {
        propagate(0, 1)
        propagate(0, 3)
    }

    2)

    1abc --- 2abc       HEAP: propagate(0, 1), propagate(0, 3), collapse(1), collapse(2), collapse(3)
    |        |
    |        |
    0a ----- 3abc

    Call: HEAP --> propagate(0, 1) PUSH TO {
        propagate(1, 2)
        propagate(1, 0)
    }

    3)

    1ab --- 2abc       HEAP: propagate(1, 0), propagate(1, 2), propagate(0, 3), collapse(1), collapse(2), collapse(3)
    |        |
    |        |
    0a ----- 3abc

    call HEAP --> propogate(1, 0) PUSH TO { }

    4)

    1ab --- 2abc       HEAP: propagate(1, 2), propagate(0, 3), collapse(1), collapse(2), collapse(3)
    |        |
    |        |
    0a ----- 3abc

    call HEAP --> propagate(1, 2) PUSH TO {
        propagate(2, 3)
        propagate(2, 1)
    }

    5)

    1ab ---- 2ac       HEAP: propagate(2, 3), propagate(2, 1), propagate(0, 3), collapse(1), collapse(2), collapse(3)
    |        |
    |        |
    0a ----- 3abc

    call HEAP --> propagate(2, 3) PUSH TO {
        propagate(3, 2)
        propagate(3, 0)
    }


    1. Find a node with the lowest entropy and collapse that node
    2. Look up each edge FROM the collapsed node in the graph's edges property
    3. For each node that the collapsed node connected TO run a union of the remaining possibilities
       and update that node's possible values
    4. Repeat step 2 & 3 propagating out until all propagations are completed
    5. Repeat from step 1
    6. Failure: Node collapses to an empty set 😿, Succcess: Heap empty 👍

    Set up master set of fully collapsed nodes,

    struct collapse {
        entropy: i32 <-- Supply any propagation method
    }

    
    OPTION NO
    struct entropoObject {
        entropy: f32
    }

    implements propgate() and collapse()

    wrapped in an enum

    Collapse(0)
    graph.constrain(nodeIndex, singletonSet) -> bool
     graph.constrain(0, {a})
     (0 {ab})

    entropy

    Sorted Collapse List { collapse(0), collapse(1) }
    Propagate Stack


    1. Start with a binary HEAP of collapses for the number of nodes in the output graph
    2. Pop the collapse action with the lowest entropy off the top of the HEAP and execute it
    3. The collapse changes the node's value and triggers a propagate on that node:
        a. Node looks up its neighbours in the edges matrix
        b. Sends a constrain message to each connected node with its set of values
        c. Its neighbours execute the constrain message with that value cross referenced with the rules
        d. If the node's value changed: 
            it generates a new collapse action and adds it to the HEAP 
            AND triggers further propagates
    4. When all propagations are complete return to step 2.
    5. When all nodes are collapsed return.



    1. Start with a binary HEAP of collapses for the number of nodes in the output graph
    2. Pop the collapse action with the lowest entropy off the top of the HEAP and execute it
    3. IF the collapse changes the node's value: 
        Triggers a propagation:
        a. Looks up its neighbours for each one it will update propagationMsgs:
            IF the nodeIndex key does not exist in the propagationMsgs Map:
                Add a entry to propagationMsgs with key=nodeIndex with value=intersection(this values, neighbour values)
            ELSE
                Update propagationMsgs entry with insection(this values, entry values)
        b. Run through the message stack and place the update set of values back into the graph.nodes
        c. graph.update(nodeIndex, set) {
            IF changed:
              propagate
        }


        [(2, 1), (2, 3)]

        propagationMsgs = HashMap (
            0: {1, 2} <- update as we go
            2: vec![{2}, {1, 2}, {2}] <- collect then intersect
        )





        c. Its neighbours execute the constrain message with that value cross referenced with the rules
        d. If the node's value changed: 
            it generates a new collapse action and adds it to the HEAP 
            AND triggers further propagates
    4. When all propagations are complete return to step 2.
    5. When all nodes are collapsed return.

    struct nodes {
        connections: HashSet,
        values: HashSet
    }

    ==== METHOD ONE ====
    HEAP: BinaryHeap
    COLLAPSE_ACTION: represents a snapshot of node's entropy for collapse
    1. Create a new binary HEAP
    2. Create a number of COLLAPSE_ACTIONs equal to the number of nodes onto the HEAP
    3. Pop the COLLAPSE_ACTION with the lowest entropy off the HEAP.
    4. Run the COLLAPSE_ACTION: 
        1. Collapse the nodeValue of the nodeIndex this references to a singleton set
        2. IF the nodeValue at nodeIndex changed:
            1. Find nodes connected to this.nodeIndex using the out_graph edges property
            2. For each connected node push the nodeValue of this and the node it effect 
               onto the PROPAGATION_MESSAGES list.
    5. Run through PROPAGATION_MESSAGES list and update the node it refers to.
    6. IF the nodeValue at nodeIndex changed:
        1. Add the nodeIndex to CHANGED_NODES list
    7. Empty PROPAGATION_MESSAGES
    8. FOR EACH node in CHANGE_NODES:
        1. Find nodes connected to this.nodeIndex using the out_graph edges property
        2. For each connected node push the nodeValue of this and the node it effect 
               onto the PROPAGATION_MESSAGES list.
    9. Repeat from 5 until CHANGED_NODES is empty


    ==== METHOD TWO ====
    HEAP: BinaryHeap
    TO_COLLAPSE: HashSet
    1. Create a new binary HEAP.
    2. Create a number of COLLAPSE_ACTION's equal to the number of nodes onto the HEAP.
    3. Peek an Action on the HEAP.
    4. IF Action == COLLAPSE_ACTION:
        1. IF TO_COLLAPSE is not empty:
            1. Generate COLLAPSE_ACTION's FOR EACH nodeIndex in TO_COLLAPSE and empty TO_COLLAPSE set.
        2. Pop an Action off of the HEAP
        3. Collapse the nodeValue of the nodeIndex this references to a singleton set.
        4. IF the nodeValue at nodeIndex changed:
            2. Find nodes connected to this.nodeIndex using the out_graph edges property
            3. For each connected node, push a PROPAGATE_ACTION to the HEAP.
        5. GOTO 3.
       ELSE IF Action == PROPAGATE_ACTION:
        1. Pop an Action off of the HEAP.
        2. Constrain nodeValue according to propagation.
        3. IF the nodeValue at nodeIndex changed:
            1. Add nodeIndex to TO_COLLAPSE.
            2. Find nodes connected to this.nodeIndex using the out_graph edges property
            3. For each connected node, push a PROPAGATE_ACTION to the HEAP.
        4. GOTO 3.
       ElSE IF HEAP is empty:
        1. return.


    EXAMPLE:

    -- Start --
    HEAP = [ Collapse(0), Collapse(1) ]
    GRAPH: 0ab --- 1ab
    TO_COLLAPSE = [ ]

    -- Loop 1 --
    PEAK Action == Collapse

    POP Action:
        TO_COLLAPSE is empty:
            no collapses generated
        Action = Collapse(0)
        HEAP = [ Collapse(1) ]

    RUN Collapse(0):
        nodeValues 0 = {a}
        connected_nodes = {1}
        HEAP = [ Propagate(1), Collapse(1) ]
        GRAPH: 0a --- 1ab
        TO_COLLAPSE = [ ]

    GOTO 3.

    -- Loop 2 --
    PEAK Action == Propagate

    POP Action

    RUN Propagate(1):
        nodeValues 1 = {}



    3. Pop the COLLAPSE_ACTION with the lowest entropy off the HEAP.
    4. Run the COLLAPSE_ACTION:
        1. Collapse the nodeValue of the nodeIndex this references to a singleton set
        2. IF the nodeValue at nodeIndex changed:
            1. Find nodes connected to this.nodeIndex using the out_graph edges property
            2. For each connected node, push a PROPAGATE_ACTION to the HEAP.




























    2. Pop the NodeCollapseAction with the lowest entropy off of the HEAP.
    3. Run a collapse on the Node that the NodeCollapseAction refers to, collapsing it to a singleton set.
    4. IF the Node's set of values changed:
        1. Find the nodes IT is connected to through out_graph edges property.
        2. Push IT's current value and the index of the node it is connected to the PropagationMsgs list.
        3. when all propagations are done and sets have been calculated run through each node on the messages
            and update that nodes values, if the node changes add to a list of changed nodes
        4. Empty PropagatioMsgs
        5. Run through each of the nodes in changed nodes which generates PropagationMsgs
        6. Empty changed nodes
        7. Run PropagationMsgs until all
    */
}

fn collapse(rules: Rules, out_graph: &Graph) {

}

#[derive(Debug)]
struct ConstraintAction {
    entropy: f32,
    to: i32,
    from: i32,
    direction: u16
}

impl ConstraintAction {
    fn new(entropy: f32, from: i32, to: i32, direction: u16) -> ConstraintAction {
        ConstraintAction {
            entropy,
            from,
            to,
            direction
        }
    }

    fn constrain(&self, graph: &Graph) {
        // code to do collapse or propagate
        
    }
}

impl Ord for ConstraintAction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.entropy.partial_cmp(&other.entropy).unwrap()
    }
}

impl Eq for ConstraintAction {}

impl PartialOrd for ConstraintAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ConstraintAction {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
    }
}

/*
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
the value that has been propagated/collapsed, then go to the graph's edges, iterate through it and look
in each value in the edges map for hashsets that contain the desired "from" node as their first value
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

Then store as sparse matrix and query directly to get node connection AND direction information immediately.

*/
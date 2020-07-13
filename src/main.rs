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
use std::collections::{HashSet, HashMap, BinaryHeap};
use std::iter::FromIterator;
use std::hash::Hash;
use std::cmp::{Eq, Ordering};
use std::clone::Clone;

type NodeValue = i32;           // values that a node can contain
type NodeIndex = i32;           // each unique node in a graph
type EdgeDirection = i32;       // the directional relationship between two nodes
type Rules = HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>>;

/*
Each node in a graph has an index,
There is a directional relationship between nodes
Each node has a value
*/

fn hash_set<T: Hash + Eq + Clone>(data: &[T]) -> HashSet<T> {
    HashSet::from_iter(data.iter().cloned())
}

fn hash_map<K: Hash + Eq + Clone, V: Clone>(data: &[(K, V)]) -> HashMap<K, V> {
    HashMap::from_iter(data.iter().cloned())
}

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
    let input_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
        [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
    );

    let input_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
        (0, hash_set(&[(0, 1), (3, 2)])), 
        (1, hash_set(&[(1, 0), (2, 3)])),
        (2, hash_set(&[(1, 2), (0, 3)])),
        (3, hash_set(&[(2, 1), (3, 0)]))
    ]);

    // generate the rules
    // Set up uncollapsed output values
    let rules: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>>;

    // generate output graph from input graph

    let output_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
        (0, hash_set(&[(0, 1), (3, 2)])), 
        (1, hash_set(&[(1, 0), (2, 3)])),
        (2, hash_set(&[(1, 2), (0, 3)])),
        (3, hash_set(&[(2, 1), (3, 0)]))
    ]);

    let input_graph = Graph {
        nodes: input_graph_nodes,
        edges: input_graph_edges
    };

    let output_graph_nodes = vec![hash_set(&[0, 1, 2]); 4];
    
    let output_graph = Graph {
        nodes: output_graph_nodes,
        edges: output_graph_edges
    };

    println!("{:?}", output_graph);

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

#[derive(Debug)]
struct EntropyAction {
    entropy: f32,
    to: i32,
    from: i32
}

impl EntropyAction {
    fn new(entropy: f32) -> EntropyAction {
        EntropyAction {
            entropy: entropy,
            to: 1,
            from: 0
        }
    }

    fn constraint(&self, graph: &Graph) {
        // code to do collapse or propagate
    }
}

impl Ord for EntropyAction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.entropy.partial_cmp(&other.entropy).unwrap()
    }
}

impl Eq for EntropyAction {}

impl PartialOrd for EntropyAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EntropyAction {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
    }
}

/*
1. Construct rules hash map
2. loop through graph directed edges
2. for each edge:
    try add (dir, NodeValue) to rules, if already in rules, union set values
*/
fn make_rules(graph: &Graph) -> Rules {
    let mut rules: Rules = HashMap::new();
    for (direction, edges) in graph.edges.iter() {
        for (from_node_index, to_node_index) in edges.iter() {
            for node_value in graph.nodes[*from_node_index as usize].iter() {
                let rules_key = (*direction, *node_value);
                let mut new_set: HashSet<i32> = HashSet::new();
                {
                    new_set.extend(rules.get(&rules_key).unwrap_or(&HashSet::new()));
                    new_set.extend(&graph.nodes[*to_node_index as usize]);
                }
                rules.insert(rules_key, new_set);
            }
        }
    }
    rules
}

#[derive(Debug)]
struct Graph {
    nodes: Vec<HashSet<NodeValue>>,
    edges: HashMap<EdgeDirection, HashSet<(i32, i32)>>
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_make_rules() {
        /*
        1b --- 2c
        |      |
        0a --- 3b
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );
    
        let test_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
            (0, hash_set(&[(0, 1), (3, 2)])), 
            (1, hash_set(&[(1, 0), (2, 3)])),
            (2, hash_set(&[(1, 2), (0, 3)])),
            (3, hash_set(&[(2, 1), (3, 0)]))
        ]);

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: test_graph_edges
        };

        // (0: N, 0: a) -> (1: b)
        // (0: N, 1: b) -> (2: c)
        // (1: S, 1: b) -> (0: a)
        // (1: S, 2: c) -> (1: b)
        // (2: E, 0: a) -> (1: b)
        // (2: E, 1: b) -> (2: c)
        // (3: W, 1: b) -> (0: a)
        // (3: W, 2: c) -> (1: b)
        
        let result: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>> = hash_map(&[
            ((0, 0), hash_set(&[1])),
            ((0, 1), hash_set(&[2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[1])),
            ((2, 0), hash_set(&[1])),
            ((2, 1), hash_set(&[2])),
            ((3, 1), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
        ]);
        
        assert_eq!(make_rules(&test_graph), result);
    }

    #[test]
    fn test_make_rules_multiple() {
        /*
        1b---2c
        |    |
        0a---3a
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
            [0, 1, 2, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );
    
        let test_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
            (0, hash_set(&[(0, 1), (3, 2)])), 
            (1, hash_set(&[(1, 0), (2, 3)])),
            (2, hash_set(&[(1, 2), (0, 3)])),
            (3, hash_set(&[(2, 1), (3, 0)]))
        ]);

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: test_graph_edges
        };

        /*
        (0: N, 0: a) -> (1: b, 2: c)
        (1: S, 1: b) -> (0: a)
        (1: S, 2: c) -> (0: a)
        (2: E, 0: a) -> (0: a)
        (2: E, 1: b) -> (2: c)
        (3: W, 0: a) -> (0: a)
        (3: W, 2: c) -> (1: b)
        */

        let result: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>> = hash_map(&[
            ((0, 0), hash_set(&[1, 2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[0])),
            ((2, 0), hash_set(&[0])),
            ((2, 1), hash_set(&[2])),
            ((3, 0), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
        ]);

        assert_eq!(make_rules(&test_graph), result);
    }

    #[test]
    fn test_make_rules_partially_collapsed() {
        /*
        1b ---- 2c
        |       |
        0ab --- 3a
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = vec![
            hash_set(&[0, 1]),
            hash_set(&[1]),
            hash_set(&[2]),
            hash_set(&[0])
        ];
    
        let test_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
            (0, hash_set(&[(0, 1), (3, 2)])), 
            (1, hash_set(&[(1, 0), (2, 3)])),
            (2, hash_set(&[(1, 2), (0, 3)])),
            (3, hash_set(&[(2, 1), (3, 0)]))
        ]);

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: test_graph_edges
        };

        /*
        (0: N, 0: a) -> (1: b, 2: c)
        (0: N, 1: b) -> (1: b)
        (1: S, 1: b) -> (0: a, 1: b)
        (1: S, 2: c) -> (0: a)
        (2: E, 0: a) -> (0: a)
        (2: E, 1: b) -> (0: a, 2: c)
        (3: W, 0: a) -> (0: a, 1: b)
        (3: W, 2: c) -> (1: b)
        */

        let result: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>> = hash_map(&[
            ((0, 0), hash_set(&[1, 2])),
            ((0, 1), hash_set(&[1])),
            ((1, 1), hash_set(&[0, 1])),
            ((1, 2), hash_set(&[0])),
            ((2, 0), hash_set(&[0])),
            ((2, 1), hash_set(&[0, 2])),
            ((3, 0), hash_set(&[0, 1])),
            ((3, 2), hash_set(&[1])),
        ]);

        assert_eq!(make_rules(&test_graph), result);
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
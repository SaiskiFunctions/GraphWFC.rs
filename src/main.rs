use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use std::hash::Hash;
use std::cmp::Eq;
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

}

fn make_rules() -> Rules {
    HashMap::new()
}

struct Graph {
    nodes: Vec<HashSet<NodeValue>>,
    edges: HashMap<EdgeDirection, HashSet<(i32, i32)>>
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_make_rules() {
        let input_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );
    
        let input_graph_edges: HashMap<EdgeDirection, HashSet<(NodeIndex, NodeIndex)>> = hash_map(&[
            (0, hash_set(&[(0, 1), (3, 2)])), 
            (1, hash_set(&[(1, 0), (2, 3)])),
            (2, hash_set(&[(1, 2), (0, 3)])),
            (3, hash_set(&[(2, 1), (3, 0)]))
        ]);

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
        
        assert_eq!(make_rules(), result);
    }
}
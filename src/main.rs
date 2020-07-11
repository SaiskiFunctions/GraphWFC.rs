use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use std::hash::Hash;
use std::cmp::Eq;
use std::clone::Clone;

type NodeValue = i32; // encoding values that a node can contain
type NodeIndex = i32; // encoding each unique node in a graph
type EdgeDirection = i32; // 

fn hash_set<T: Hash + Eq + Clone>(data: &[T]) -> HashSet<T> {
    HashSet::from_iter(data.iter().cloned())
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

    let input_graph_edges = [
        (0, [(0, 1), (3, 2)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (1, [(1, 0), (2, 3)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (2, [(1, 2), (0, 3)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (3, [(2, 1), (3, 0)].iter().cloned().collect::<HashSet<(i32, i32)>>())
        ].iter().cloned().collect::<HashMap<EdgeDirection, HashSet<(i32, i32)>>>();

    // generate the rules
    // Set up uncollapsed output values
    let rules: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>>;
    // (0: N, 0: a) -> (1: b)
    // (0: N, 1: b) -> (2: c)
    // (1: S, 1: b) -> (0: a)
    // (1: S, 2: c) -> (1: b)
    // (2: E, 0: a) -> (1: b)
    // (2: E, 1: b) -> (2: c)
    // (3: W, 1: b) -> (0: a)
    // (3: W, 2: c) -> (1: b)
    
    // 

    // generate output graph from input graph

    let output_graph_edges = [
        (0, hash_set(&[(1, 2), (4, 3)])), 
        (1, hash_set(&[(2, 1), (3, 4)])),
        (2, hash_set(&[(2, 3), (1, 4)])),
        (3, hash_set(&[(3, 2), (4, 1)]))
        ].iter().cloned().collect::<HashMap<EdgeDirection, HashSet<(i32, i32)>>>();

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

fn make_rules() {

}

struct Graph {
    nodes: Vec<HashSet<NodeValue>>,
    edges: HashMap<EdgeDirection, HashSet<(i32, i32)>>
}

#[cfg(test)]
mod tests {

    #[test]
    fn passes() {
        assert_eq!(2, 2);
    }
}
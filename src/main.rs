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
}
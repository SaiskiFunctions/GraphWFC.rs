use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use crate::utils::{hash_map, hash_set};

pub type NodeValue = i32;           // values that a node can contain
pub type NodeIndex = i32;           // each unique node in a graph
pub type EdgeDirection = u16;       // the directional relationship between two nodes
pub type Edges = HashMap<NodeIndex, Vec<(NodeIndex, EdgeDirection)>>;
pub type Rules = HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>>;

#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<HashSet<NodeValue>>,
    edges: Edges
}

impl Graph {
    pub fn new(nodes: Vec<HashSet<NodeValue>>, edges: Edges) -> Graph {
        Graph {
            nodes,
            edges
        }
    }

    pub fn constrain(&mut self, index: NodeIndex, values: HashSet<NodeValue>) {
        self.nodes.get_mut(index as usize).map(|set| {
            set.intersection(&values).collect::<HashSet<&NodeValue>>()
        });
    }

    /*
    1. Construct rules hash map
    2. loop through graph directed edges
    2. for each edge:
        try add (dir, NodeValue) to rules, if already in rules, union set values
    */
    pub fn make_rules(&self) -> Rules {
        let mut rules: Rules = HashMap::new();
        for (from_node_index, edges) in self.edges.iter() {
            for (to_node_index, direction) in edges.iter() {
                for node_value in self.nodes[*from_node_index as usize].iter() {
                    let rules_key = (*direction, *node_value);
                    rules.entry(rules_key)
                        .and_modify(|set| set.extend(&self.nodes[*to_node_index as usize]))
                        .or_insert(self.nodes[*to_node_index as usize].clone());
                }
            }
        }
        rules
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    #[test]
    fn test_make_rules() {
        /*
        1b --- 2c
        |      |
        0a --- 3b

        North = 0, South = 1, East = 2, West = 3
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: graph_edges()
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

        assert_eq!(test_graph.make_rules(), result);
    }

    #[test]
    fn test_make_rules_multiple() {
        /*
        1b---2c
        |    |
        0a---3a

        North = 0, South = 1, East = 2, West = 3
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = Vec::from_iter(
            [0, 1, 2, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: graph_edges()
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

        assert_eq!(test_graph.make_rules(), result);
    }

    #[test]
    fn test_make_rules_partially_collapsed() {
        /*
        1b ---- 2c
        |       |
        0ab --- 3a

        North = 0, South = 1, East = 2, West = 3
        */

        let test_graph_nodes: Vec<HashSet<NodeValue>> = vec![
            hash_set(&[0, 1]),
            hash_set(&[1]),
            hash_set(&[2]),
            hash_set(&[0])
        ];

        let test_graph = Graph {
            nodes: test_graph_nodes,
            edges: graph_edges()
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

        assert_eq!(test_graph.make_rules(), result);
    }
}

use std::collections::{HashSet, HashMap};
type NodeValue = i32;
type EdgeDirection = i32;

fn main() {

    // Preamble
    let node_values = [0, 1, 2].iter().cloned().collect::<HashSet<NodeValue>>();
    // {0, 1, 2}

    let rules: HashMap<(EdgeDirection, NodeValue), HashSet<NodeValue>>;

    /*
    2b --- 3c
    |      |
    1a --- 4b
    North = 0, South = 1, East = 2, West = 3
    */
    let output_graph_edges = [
        (0, [(1, 2), (4, 3)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (1, [(2, 1), (3, 4)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (2, [(2, 3), (1, 4)].iter().cloned().collect::<HashSet<(i32, i32)>>()),
        (3, [(3, 2), (4, 1)].iter().cloned().collect::<HashSet<(i32, i32)>>())
        ].iter().cloned().collect::<HashMap<EdgeDirection, HashSet<(i32, i32)>>>();

    let output_graph_nodes = [1, 2, 3, 4].iter().cloned().collect::<HashSet<i32>>();
    
    let output_graph = Graph {
        nodes: output_graph_nodes,
        edges: output_graph_edges
    };

    // Set up uncollapsed output values
    let diagonal = vec![node_values; output_graph.nodes.len()];
}



struct Graph {
    nodes: HashSet<i32>,
    edges: HashMap<EdgeDirection, HashSet<(i32, i32)>>
}

#[cfg(test)]
mod tests {

    #[test]
    fn passes() {
        assert_eq!(2, 2);
    }
}
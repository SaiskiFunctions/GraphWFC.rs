use std::collections::{HashSet, BinaryHeap, HashMap};
use std::mem::replace;
use std::ops::Index;
use rand::prelude::*;
use crate::observe::Observe;
use crate::propagate::Propagate;
use crate::graph::{Rules, Graph, VertexIndex, VertexLabel, Frequencies, Labels};
use crate::utils::{hash_set, hash_map};

struct Collapse<'a> {
    heap: BinaryHeap<Observe>,
    gen_observe: HashSet<VertexIndex>,
    observed: HashSet<VertexIndex>,
    propagations: Vec<Propagate>,
    rng: &'a mut StdRng,
    rules: &'a Rules,
    frequencies: &'a Frequencies,
    all_labels: &'a Labels,
    out_graph: Graph,
}

impl Collapse<'_> {
    pub fn new<'a>(rng: &'a mut StdRng,
                   rules: &'a Rules,
                   frequencies: &'a Frequencies,
                   all_labels: &'a Labels,
                   out_graph: Graph) -> Collapse<'a> {
        let mut heap: BinaryHeap<Observe> = BinaryHeap::new();
        let mut gen_observe: HashSet<VertexIndex> = HashSet::new();
        let mut observed: HashSet<VertexIndex> = HashSet::new();
        let mut propagations: Vec<Propagate> = Vec::new();

        for (_index, labels) in out_graph.vertices.iter().enumerate() {
            let from_index = _index as i32;
            if labels.is_subset(all_labels) && labels != all_labels { // <-- labels is proper subset of all_labels
                out_graph.connections(&from_index).iter().for_each(|(to_index, direction)| {
                    propagations.push(Propagate::new(from_index, *to_index, *direction))
                });
                if labels.len() == 1 { // <-- labels is singleton set
                    observed.insert(from_index);
                    continue;
                }
            }
            heap.push(Observe::new_fuzz(rng, &from_index, labels, frequencies))
        }

        Collapse {
            heap,
            gen_observe,
            observed,
            propagations,
            rng,
            rules,
            frequencies,
            all_labels,
            out_graph
        }
    }

    pub fn exec(&mut self) -> Option<Graph> {
        let heap = &mut self.heap;
        let observed = &mut self.observed;
        let out_graph = &mut self.out_graph;
        let gen_observe = &mut self.gen_observe;
        let propagations = &mut self.propagations;

        let frequencies = &self.frequencies;
        let vertices = &out_graph.vertices;

        loop {
            if observed.len() == vertices.len() || heap.is_empty() {
                return Some(replace(out_graph, Graph::empty()));
            }
            if propagations.is_empty() {
                gen_observe.drain().for_each(|index| {  // algo: 4.2
                    let labels = vertices.index(index as usize);
                    heap.push(Observe::new(&index, labels, frequencies))
                });
                //heap.pop().unwrap()
            } else {
                // do propagate
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::graph::EdgeDirection;

    #[test]
    fn test_new() {
        let mut rng = StdRng::seed_from_u64(10);

        let edges = hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ]);

        let graph_vertices: Vec<HashSet<VertexLabel>> = vec![
            hash_set(&[0, 1, 2]),
            hash_set(&[0, 1, 2]),
            hash_set(&[0, 1, 2]),
            hash_set(&[0, 1, 2])
        ];

        let out_graph = Graph::new(graph_vertices, edges);

        let rules: HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>> = hash_map(&[
            ((0, 0), hash_set(&[1])),
            ((0, 1), hash_set(&[2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[1])),
            ((2, 0), hash_set(&[1])),
            ((2, 1), hash_set(&[2])),
            ((3, 1), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
        ]);

        let frequencies = hash_map(&[(0, 1), (1, 2), (2, 1)]);

        let all_labels = hash_set(&[0, 1, 2]);


        let collapse = Collapse::new(&mut rng,
                   &rules,
                   &frequencies,
                   &all_labels,
                   out_graph);

        assert_eq!(collapse.heap.len(), 4);
    }

    #[test]
    fn test_new_partial() {
        let mut rng = StdRng::seed_from_u64(10);

        let edges = hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ]);

        let graph_vertices: Vec<HashSet<VertexLabel>> = vec![
            hash_set(&[0]),
            hash_set(&[0, 2]),
            hash_set(&[0, 1, 2]),
            hash_set(&[0, 1, 2])
        ];

        let out_graph = Graph::new(graph_vertices, edges);

        let rules: HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>> = hash_map(&[
            ((0, 0), hash_set(&[1])),
            ((0, 1), hash_set(&[2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[1])),
            ((2, 0), hash_set(&[1])),
            ((2, 1), hash_set(&[2])),
            ((3, 1), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
        ]);

        let frequencies = hash_map(&[(0, 1), (1, 2), (2, 1)]);

        let all_labels = hash_set(&[0, 1, 2]);


        let collapse = Collapse::new(&mut rng,
                   &rules,
                   &frequencies,
                   &all_labels,
                   out_graph);

        assert_eq!(collapse.heap.len(), 3);
        assert_eq!(collapse.propagations.len(), 4);
        assert_eq!(collapse.observed, hash_set(&[0]));
    }
}
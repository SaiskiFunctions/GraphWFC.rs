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

        let vertices_iter = out_graph.vertices.iter().enumerate();
        
        // initialize heap and observed
        for (_index, labels) in vertices_iter.clone() {
            let from_index = _index as i32;
            if labels.len() == 1 { // <-- labels is singleton set
                observed.insert(from_index);
                continue
            }
            heap.push(Observe::new_fuzz(rng, &from_index, labels, frequencies))
        }

        // initialize propagates
        for (_index, labels) in vertices_iter {
            let from_index = _index as i32;
            if labels.is_subset(all_labels) && labels != all_labels { // <-- labels is proper subset of all_labels
                generate_propagations(&mut propagations, &observed, &out_graph, &from_index);
            }
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
        let rules = &self.rules;
        //let vertices = &mut out_graph.vertices;

        loop {
            if observed.len() == out_graph.vertices.len() || heap.is_empty() {
                return Some(replace(out_graph, Graph::empty()));
            }
            if propagations.is_empty() {
                gen_observe.drain().for_each(|index| {  // algo: 4.2
                    let labels = out_graph.vertices.index(index as usize);
                    heap.push(Observe::new(&index, labels, frequencies))
                });
                
                let observe = heap.pop().unwrap();
                if !observed.contains(&observe.index) {
                    out_graph.observe(self.rng, &observe.index, frequencies);

                    generate_propagations(propagations, observed, out_graph, &observe.index);
                }
            } else {
                let propagate = propagations.pop().unwrap();

                let constraint = out_graph.vertices.index(propagate.from as usize).iter()
                .fold(HashSet::new(), |mut cst, label| {
                    cst.extend(rules.index(&(propagate.direction, *label)));
                    cst
                });

                if let Some(labels) = out_graph.constrain(&propagate.to, &constraint) { //🎸
                    if labels.is_empty() {
                        return None
                    } else if labels.len() == 1 {
                        // add to observe
                        observed.insert(propagate.to);
                    } else {
                        // add to gen
                        gen_observe.insert(propagate.to);
                    }

                    generate_propagations(propagations, observed, out_graph, &propagate.to);
                }
            }
        }
    }
}

fn generate_propagations(propagations: &mut Vec<Propagate>, observed: &HashSet<VertexIndex>, out_graph: &Graph, from_index: &VertexIndex) {
    out_graph.connections(from_index).iter().for_each(|(to_index, direction)| {
        if !observed.contains(to_index) {
            propagations.push(Propagate::new(*from_index, *to_index, *direction))
        }
    });
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
        assert_eq!(collapse.propagations.len(), 3);
        assert_eq!(collapse.observed, hash_set(&[0]));
    }
}
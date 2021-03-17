use crate::graph::graph::{EdgeDirection, Edges, Graph, Rules, VertexIndex};
use crate::wfc::observe::Observe;
use crate::wfc::propagate::Propagate;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::BinaryHeap;
use std::mem::replace;
use std::ops::{Index, IndexMut};
use crate::utils::Metrics;
use bit_set::BitSet;
use crate::MSu16xNU;

type InitCollapse = (
    BitSet,               // observed
    Vec<Propagate>,       // propagations
    Vec<VertexIndex>,     // to_observe
    BinaryHeap<Observe>,  // heap
);

fn init_collapse(rng: &mut SmallRng, out_graph: &Graph) -> InitCollapse {
    let mut observed: BitSet = BitSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();
    let mut init_propagations: Vec<VertexIndex> = Vec::new();
    let mut heap: BinaryHeap<Observe> = BinaryHeap::new();

    // initialize heap and observed
    out_graph
        .vertices
        .iter()
        .enumerate()
        .for_each(|(_index, labels)| {
            assert!(labels.is_subset(&out_graph.all_labels));
            let from_index = _index as VertexIndex;
            if labels.is_singleton() {
                init_propagations.push(from_index);
                observed.insert(from_index as usize);
            } else if labels != &out_graph.all_labels {
                init_propagations.push(from_index);
                heap.push(Observe::new(from_index, labels.shannon_entropy()))
            }
        });

    // Ensure that output graph will be fully propagated before further collapse.
    init_propagations.drain(..).for_each(|index| {
        generate_propagations(&mut propagations, &observed, &out_graph.edges, index);
    });

    let to_observe_len = out_graph.vertices.len() as VertexIndex;
    let mut to_observe: Vec<VertexIndex> = (0..to_observe_len)
        .filter(|i| !observed.contains(*i as usize))
        .collect();
    to_observe.shuffle(rng);

    (observed, propagations, to_observe, heap)
}

const METRICS: bool = false;
const OBSERVE_CHANCE: usize = 33;

fn exec_collapse(
    rng: &mut SmallRng,
    rules: &Rules,
    edges: &Edges,
    init: InitCollapse,
    mut vertices: Vec<MSu16xNU>,
    partial: bool
) -> Vec<MSu16xNU> {
    let (mut observed, mut propagations, mut to_observe, mut heap) = init;
    let mut to_propagate: Vec<Propagate> = Vec::new();
    let vertices_len = vertices.len();
    let mut observed_counter: usize = 0;

    let mut metrics = Metrics::new();

    if METRICS {
        metrics.avg("props/obs", ("props", "obs"));
        metrics.avg("props/loops", ("props", "loops"));
    }

    loop {
        // propagate constraints
        while !propagations.is_empty() {
            if METRICS { metrics.inc("loops") }

            for propagate in propagations.drain(..) {
                if METRICS { metrics.inc("props") }

                assert!(vertices.len() >= propagate.from as usize);
                let prop_labels = vertices.index(propagate.from as usize);
                if prop_labels.is_empty() {
                    continue
                }

                let constraint = build_constraint(prop_labels, propagate.direction, rules);

                assert!(vertices.len() >= propagate.to as usize);
                let labels = vertices.index_mut(propagate.to as usize);

                let constrained = labels.intersection(&constraint);
                if labels != &constrained {
                    if constrained.is_singleton() || constrained.is_empty() {
                        observed.insert(propagate.to as usize);
                        observed_counter += 1
                    } else if rng.gen_range(0..100) < OBSERVE_CHANCE {
                        let entropy = constrained.shannon_entropy();
                        println!("{}", entropy);
                        heap.push(Observe::new(propagate.to, entropy))
                    }
                    generate_propagations(&mut to_propagate, &observed, edges, propagate.to);
                    *labels = constrained
                }
            }
            to_propagate = replace(&mut propagations, to_propagate);
        }

        if partial { return vertices }

        // check if all vertices observed, if so we have finished
        if observed_counter == vertices_len {
            if METRICS { metrics.print(Some("All Observed")) }
            return vertices;
        }

        // try to find a vertex index to observe
        let mut observe_index: Option<VertexIndex> = None;
        // check the heap first
        while !heap.is_empty() {
            let observe = heap.pop().unwrap();
            if !observed.contains(observe.index as usize) {
                observe_index = Some(observe.index);
                break;
            }
        }
        // if no index to check in heap, check vec of initial vertices to observe
        if observe_index.is_none() {
            while !to_observe.is_empty() {
                let index = to_observe.pop().unwrap();
                if !observed.contains(index as usize) {
                    observe_index = Some(index);
                    break;
                }
            }
        }
        match observe_index {
            None => {
                if METRICS { metrics.print(Some("All Observed")) }
                // Nothing left to observe, therefore we've finished
                return vertices;
            }
            Some(index) => {
                if METRICS { metrics.inc("obs") }

                assert!(vertices.len() >= index as usize);
                let labels_multiset = vertices.index_mut(index as usize);
                labels_multiset.choose_random(rng);
                observed.insert(index as usize);
                generate_propagations(&mut propagations, &observed, edges, index);
            }
        }
    }
}

pub fn build_constraint(labels: &MSu16xNU, direction: EdgeDirection, rules: &Rules) -> MSu16xNU {
    labels
        .into_iter()
        .enumerate()
        .fold(MSu16xNU::empty(), |mut acc, (index, frequency)| {
            if frequency > 0 {
                if let Some(a) = rules.get(&(direction, index)) {
                    acc = acc.union(a)
                }
            }
            acc
        })
}

fn generate_propagations(
    propagations: &mut Vec<Propagate>,
    observed: &BitSet,
    edges: &Edges,
    from_index: VertexIndex,
) {
    assert!(edges.contains_key(&from_index));
    for (to_index, direction) in edges.index(&from_index) {
        if !observed.contains(*to_index as usize) {
            propagations.push(Propagate::new(from_index, *to_index, *direction))
        }
    }
}

pub fn collapse(
    rules: &Rules,
    mut output_graph: Graph,
    seed: Option<u64>,
    partial: bool
) -> Graph {
    let rng = &mut SmallRng::seed_from_u64(seed.unwrap_or_else(|| thread_rng().next_u64()));
    let init = init_collapse(rng, &output_graph);

    let collapsed_vertices = exec_collapse(
        rng,
        rules,
        &output_graph.edges,
        init,
        output_graph.vertices,
        partial // 🐯
    );
    output_graph.vertices = collapsed_vertices;
    output_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Edges;
    use crate::utils::hash_map;
    use std::iter::FromIterator;

    //noinspection DuplicatedCode
    fn simple_vertices() -> Vec<MSu16xNU> {
        vec![
            [1, 0, 0].iter().collect(),
            [0, 2, 0].iter().collect(),
            [0, 0, 1].iter().collect(),
            [0, 2, 0].iter().collect(),
        ]
    }

    #[test]
    fn test_constraint() {
        let rules: Rules = hash_map(&[
            ((0, 0), [1, 0, 0, 0, 0, 0].iter().collect()),
            ((0, 1), [0, 0, 1, 0, 0, 0].iter().collect()),
            ((0, 2), [1, 1, 1, 0, 0, 0].iter().collect())
        ]);

        let labels = [2, 4, 0, 0, 0, 0].iter().collect();
        let direction: EdgeDirection = 0;

        let result = build_constraint(&labels, direction, &rules);
        let expected: MSu16xNU = [1, 0, 1, 0, 0, 0].iter().collect();
        assert_eq!(result, expected)
    }

    #[test]
    fn test_constraint2() {
        let rules: Rules = hash_map(&[
            ((0, 0), [2, 0, 0, 0, 0, 0].iter().collect()),
            ((0, 1), [0, 0, 5, 0, 0, 0].iter().collect()),
            ((0, 2), [1, 1, 1, 0, 0, 0].iter().collect())
        ]);

        let labels = [3, 4, 1, 0, 2, 0].iter().collect();
        let direction: EdgeDirection = 0;

        let result = build_constraint(&labels, direction, &rules);
        let expected: MSu16xNU = [2, 1, 5, 0, 0, 0].iter().collect();
        assert_eq!(result, expected)
    }

    //noinspection DuplicatedCode
    #[test]
    fn test_exec_simple() {
        let mut rng = SmallRng::seed_from_u64(3);

        let edges = hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ]);
        let all_labels = MSu16xNU::from_iter([1, 2, 1].iter().cloned());
        let out_graph = Graph::new(simple_vertices(), edges, all_labels);
        let rules: Rules = hash_map(&[
            ((0, 0), [0, 2, 0].iter().collect()),
            ((0, 1), [0, 0, 1].iter().collect()),
            ((1, 1), [1, 0, 0].iter().collect()),
            ((1, 2), [0, 2, 0].iter().collect()),
            ((2, 0), [0, 2, 0].iter().collect()),
            ((2, 1), [0, 0, 1].iter().collect()),
            ((3, 1), [1, 0, 0].iter().collect()),
            ((3, 2), [0, 2, 0].iter().collect()),
        ]);

        let init = init_collapse(&mut rng, &out_graph);

        let result = exec_collapse(&mut rng, &rules, &out_graph.edges, init, simple_vertices(), false);
        let expected: Vec<MSu16xNU> = vec![
            [1, 0, 0].iter().collect(),
            [0, 2, 0].iter().collect(),
            [0, 0, 1].iter().collect(),
            [0, 2, 0].iter().collect(),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_exec_med() {
        /*
            INPUT:
            0b --- 1b --- 2a
            |      |      |
            3a --- 4b --- 5a

            Output structure same as input structure.
            North = 0, South = 1, East = 2, West = 3
        */
        let mut rng = SmallRng::seed_from_u64(246547);
        let all_labels = MSu16xNU::from_iter([3, 3].iter().cloned());

        let edges = hash_map(&[
            (0, vec![(3, 1), (1, 2)]),
            (1, vec![(0, 3), (4, 1), (2, 2)]),
            (2, vec![(1, 3), (5, 1)]),
            (3, vec![(0, 0), (4, 2)]),
            (4, vec![(3, 3), (1, 0), (5, 2)]),
            (5, vec![(4, 3), (2, 0)]),
        ]);

        let vertices: Vec<MSu16xNU> = vec![all_labels; 6];

        let rules: Rules = hash_map(&[
            ((0, 0), all_labels),
            ((0, 1), all_labels),
            ((1, 0), all_labels),
            ((1, 1), all_labels),
            ((2, 0), all_labels),
            ((2, 1), all_labels),
            ((3, 0), all_labels),
            ((3, 1), all_labels),
        ]);

        let out_graph = Graph::new(vertices, edges, all_labels);
        let init = init_collapse(&mut rng, &out_graph);

        let result = exec_collapse(
            &mut rng,
            &rules,
            &out_graph.edges,
            init,
            out_graph.vertices.clone(),
            false
        );

        let expected: Vec<MSu16xNU> = vec![
            [0, 3].iter().collect(),
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_exec_complex() {
        /*
            INPUT graph:
                   0a --- 1b --- 2b
                   |      |
            3a --- 4a --- 5b

            Directions: North = 0, South = 1, East = 2, West = 3
        */

        let mut rng = SmallRng::seed_from_u64(2);
        let all_labels = MSu16xNU::from_iter([3, 3].iter().cloned());

        let input_edges = hash_map(&[
            (0, vec![(1, 2), (4, 1)]),
            (1, vec![(0, 3), (5, 1), (2, 2)]),
            (2, vec![(1, 3)]),
            (3, vec![(4, 2)]),
            (4, vec![(3, 3), (0, 0), (5, 2)]),
            (5, vec![(4, 3), (1, 0)]),
        ]);

        let input_vertices: Vec<MSu16xNU> = vec![
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
            [0, 3].iter().collect(),
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
        ];

        let input_graph = Graph::new(input_vertices, input_edges, all_labels);

        let rules = input_graph.rules();

        /*
            OUTPUT STRUCTURE:
            0 ---- 1 ---- 2 ---- 3
            |      |      |      |
            4 ---- 5 ---- 6 ---- 7
            |      |      |      |
            8 ---- 9 ---- 10 --- 11
        */

        let output_edges: Edges = hash_map(&[
            (0, vec![(1, 2), (4, 1)]),
            (1, vec![(0, 3), (5, 1), (2, 2)]),
            (2, vec![(1, 3), (6, 1), (3, 2)]),
            (3, vec![(2, 3), (7, 1)]),
            (4, vec![(0, 0), (8, 1), (5, 2)]),
            (5, vec![(4, 3), (1, 0), (6, 2), (9, 1)]),
            (6, vec![(5, 3), (2, 0), (7, 2), (10, 1)]),
            (7, vec![(6, 3), (3, 0), (11, 1)]),
            (8, vec![(4, 0), (9, 2)]),
            (9, vec![(8, 3), (5, 0), (10, 2)]),
            (10, vec![(9, 3), (6, 0), (11, 2)]),
            (11, vec![(10, 3), (7, 0)]),
        ]);

        let output_vertices: Vec<MSu16xNU> = vec![all_labels; 12];

        let output_graph = Graph::new(output_vertices, output_edges, all_labels);
        let init = init_collapse(&mut rng, &output_graph);

        let result = exec_collapse(
            &mut rng,
            &rules,
            &output_graph.edges,
            init,
            output_graph.vertices.clone(),
            false
        );

        let expected: Vec<MSu16xNU> = vec![
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
            [0, 3].iter().collect(),
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
            [0, 3].iter().collect(),
            [3, 0].iter().collect(),
            [3, 0].iter().collect(),
            [0, 3].iter().collect(),
            [0, 3].iter().collect(),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_entropy() {
        let x: MSu16xNU = [3, 0, 1, 5, 2, 6, 1].iter().collect();
        println!("{}", x.shannon_entropy())
    }

    #[test]
    fn test_rng() {
        let mut rng = SmallRng::seed_from_u64(0);
        (0..25)
            .for_each(|_| println!("{}", rng.next_u64()))
    }
}

use rand::prelude::*;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::collections::{BinaryHeap};
use hashbrown::HashSet;
use std::mem::replace;
use std::ops::Index;
use crate::graph::graph::{VertexIndex, Rules, Graph, Edges};
use crate::wfc::observe::Observe;
use crate::wfc::propagate::Propagate;
use crate::multiset::{MultisetTrait, Multiset, MultisetScalar};
use nalgebra::{Dim, DimName, DefaultAllocator};
use nalgebra::allocator::Allocator;


fn init_collapse<D>(
    rng: &mut StdRng,
    all_labels: &Multiset<D>,
    out_graph: &mut Graph<D>,
) -> (HashSet<VertexIndex>, Vec<Propagate>, Vec<VertexIndex>)
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let mut observed: HashSet<VertexIndex> = HashSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();
    let mut init_propagations: Vec<u32> = Vec::new();

    // initialize heap and observed
    out_graph.vertices.iter().enumerate().for_each(|(_index, labels)| {
        assert!(labels.is_subset(all_labels));
        let from_index = _index as u32;
        if labels.is_singleton() { observed.insert(from_index); }
        else if labels != all_labels { init_propagations.push(from_index) }
    });

    // Ensure that output graph is fully propagated before starting loop.
    // Generate Propagates for every vertex whose label set is a proper
    // subset of the set of all labels.
    init_propagations.drain(..).for_each(|index| {
        generate_propagations(&mut propagations, &observed, &out_graph.edges, &index);
    });

    let mut to_observe: Vec<VertexIndex> = (0..out_graph.vertices.len() as u32).collect();
    to_observe.shuffle(rng);

    (observed, propagations, to_observe)
}

fn exec_collapse<D>(
    rng: &mut StdRng,
    rules: &Rules<D>,
    all_labels: &Multiset<D>,
    edges: &Edges,
    out_graph: &mut Graph<D>,
) -> Option<Graph<D>>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let (mut observed, mut propagations, mut to_observe) = init_collapse(rng, all_labels, out_graph);
    let heap: &mut BinaryHeap<Observe> = &mut BinaryHeap::new();
    let gen_observe: &mut HashSet<VertexIndex> = &mut HashSet::new();

    loop {
        if propagations.is_empty() {
            gen_observe.drain().for_each(|index| {
                assert!(out_graph.vertices.len() >= index as usize);
                let labels = out_graph.vertices.index(index as usize);
                heap.push(Observe::new(&index, labels.entropy()))
            });

            let observe_index = if let Some(ob) = heap.pop() {
                ob.index
            } else {
                let mut _index: u32;
                loop {
                    if let Some(new) = to_observe.pop() {
                        if observed.contains(&new) { continue }
                        _index = new;
                        break
                    } else {
                        return Some(replace(out_graph, Graph::empty()))
                    }
                }
                _index
            };

            if observed.contains(&observe_index) { continue }
            out_graph.observe(rng, &observe_index);
            observed.insert(observe_index);
            generate_propagations(&mut propagations, &observed, edges, &observe_index);
        } else {
            let propagate = propagations.pop().unwrap();
            assert!(out_graph.vertices.len() >= propagate.from as usize);
            let prop_labels = out_graph.vertices.index(propagate.from as usize);
            let constraint = constraint(prop_labels, &propagate.direction, rules);

            if let Some(labels) = out_graph.constrain(&propagate.to, &constraint) { //🎸
                if labels.is_empty() { return None }
                else if labels.is_singleton() { observed.insert(propagate.to); }
                else { gen_observe.insert(propagate.to); }
                generate_propagations(&mut propagations, &observed, edges, &propagate.to);
            }
        }
    }
}

pub fn constraint<D>(labels: &Multiset<D>, direction: &u16, rules: &Rules<D>) -> Multiset<D>
where D: Dim + DimName,
      DefaultAllocator: Allocator<MultisetScalar, D>
{
    labels.iter().enumerate()
        .fold(Multiset::<D>::zeros(), |mut acc, (index, frq)| {
            if frq > &0 {
                if let Some(a) = rules.get(&(*direction, index)) {
                    acc = acc.sup(a)
                }
            }
            acc
        })
}

fn generate_propagations(propagations: &mut Vec<Propagate>, observed: &HashSet<VertexIndex>, edges: &Edges, from_index: &VertexIndex) {
    assert!(edges.len() >= *from_index as usize);
    edges.index(from_index).iter().for_each(|(to_index, direction)| {
        if !observed.contains(to_index) {
            propagations.push(Propagate::new(*from_index, *to_index, *direction))
        }
    });
}

pub fn collapse<D>(input_graph: &Graph<D>, output_graph: Graph<D>, seed: Option<u64>, tries: Option<u16>) -> Option<Graph<D>>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let rng = &mut StdRng::seed_from_u64(seed.unwrap_or_else(|| {
        thread_rng().next_u64()
    }));
    let tries = tries.unwrap_or(10);
    let rules = &input_graph.rules();
    let all_labels = &input_graph.all_labels;

    for _ in 0..tries {
        let new_output = &mut output_graph.clone();
        if let opt_graph @ Some(_) = exec_collapse(rng, rules, all_labels, &output_graph.edges, new_output) {
            return opt_graph;
        }
    }
    println!("CONTRADICTION!");
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Edges;
    use crate::utils::hash_map;
    use nalgebra::{U6, U2};

    //noinspection DuplicatedCode
    fn simple_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    //noinspection DuplicatedCode
    fn simple_vertices() -> Vec<Multiset<U6>> {
        vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
        ]
    }

    //noinspection DuplicatedCode
    fn simple_rules() -> Rules<U6> {
        hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((0, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((1, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((1, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((3, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((3, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
        ])
    }

    #[test]
    fn test_constraint() {
        let a: Multiset<U6> = Multiset::from_row_slice_u(&[1, 0, 0, 0, 0, 0]);
        let b: Multiset<U6> = Multiset::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        let c: Multiset<U6> = Multiset::from_row_slice_u(&[1, 1, 1, 0, 0, 0]);
        let rules: Rules<U6> = hash_map(&[
            ((0, 0), a),
            ((0, 1), b),
            ((0, 2), c)
        ]);

        let labels: &Multiset<U6> = &Multiset::from_row_slice_u(&[2, 4, 0, 0, 0, 0]);
        let direction: &u16 = &0;

        let result = constraint(labels, direction, &rules);
        let expected: Multiset<U6> = Multiset::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        assert_eq!(result, expected)
    }

    //noinspection DuplicatedCode
    #[test]
    fn test_exec_simple() {
        /* Seed Values:
            3 -> Some([{0}, {1}, {2}, {1}])
            1 -> None
        */
        let mut rng = StdRng::seed_from_u64(3);

        let edges = simple_edges();
        let all_labels = Multiset::<U6>::from_row_slice_u(&[1, 2, 1]);
        let mut out_graph = Graph::<U6>::new(simple_vertices(), edges.clone(), all_labels);
        let rules: &Rules<U6> = &simple_rules();

        let result = exec_collapse::<U6>(&mut rng, rules, &all_labels, &edges, &mut out_graph).unwrap();
        let expected: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
        ];

        assert_eq!(result.vertices, expected);
    }

    #[test]
    fn test_exec_med() {
        /* Seed Values:
            INPUT:
            0b --- 1b --- 2a
            |      |      |
            3a --- 4b --- 5a

            Output structure same as input structure.
            North = 0, South = 1, East = 2, West = 3
        */
        let mut rng = StdRng::seed_from_u64(234);

        let edges = hash_map(&[
            (0, vec![(3, 1), (1, 2)]),
            (1, vec![(0, 3), (4, 1), (2, 2)]),
            (2, vec![(1, 3), (5, 1)]),
            (3, vec![(0, 0), (4, 2)]),
            (4, vec![(3, 3), (1, 0), (5, 2)]),
            (5, vec![(4, 3), (2, 0)])
        ]);

        let vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
        ];

        let rules: Rules<U2> = hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((0, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((1, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((1, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((2, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((2, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((3, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((3, 1), Multiset::from_row_slice_u(&[3, 3])),
        ]);

        let all_labels = Multiset::from_row_slice_u(&[3, 3]);
        let out_graph = &mut Graph::<U2>::new(vertices, edges.clone(), all_labels);

        let result = exec_collapse::<U2>(&mut rng,
                                         &rules,
                                         &all_labels,
                                         &edges,
                                         out_graph).unwrap();

        let expected: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
        ];


        assert_eq!(result.vertices, expected);
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

        let mut rng = StdRng::seed_from_u64(1);

        let input_edges = hash_map(&[
            (0, vec![(1, 2), (4, 1)]),
            (1, vec![(0, 3), (5, 1), (2, 2)]),
            (2, vec![(1, 3)]),
            (3, vec![(4, 2)]),
            (4, vec![(3, 3), (0, 0), (5, 2)]),
            (5, vec![(4, 3), (1, 0)])
        ]);

        let input_vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3])
        ];

        let all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);

        let input_graph = Graph::<U2>::new(input_vertices, input_edges, all_labels);

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
            (11, vec![(10, 3), (7, 0)])
        ]);

        let output_vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3])
        ];

        // should be same as all labels but Multiset.clone() is outputs false
        // positive errors, so recreating here to avoid.
        let out_all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);

        let output_graph = &mut Graph::<U2>::new(output_vertices, output_edges.clone(), out_all_labels);

        let result = exec_collapse::<U2>(&mut rng,
                                         &rules,
                                         &all_labels,
                                         &output_edges,
                                         output_graph).unwrap();

        let expected: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3])
        ];

        assert_eq!(result.vertices, expected);
    }
}

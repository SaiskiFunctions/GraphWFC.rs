#[macro_use]
extern crate bencher;
extern crate nalgebra;


mod collapse {
    use bencher::Bencher;
    use wfc_rust::wfc::collapse::collapse;
    use wfc_rust::io::text_parser::{parse, make_edges_cardinal_grid};
    use wfc_rust::graph::graph::{Graph, Rules};
    use wfc_rust::multiset::{Multiset, MultisetTrait};
    use wfc_rust::utils::hash_map;
    use nalgebra::{U6};

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse::<U6>("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels;
            let output_vertices: Vec<Multiset<U6>> = vec![all_labels; out_width * out_depth];
            let output_edges = make_edges_cardinal_grid(out_width, out_depth);
            let output_graph = Graph::<U6>::new(output_vertices, output_edges, all_labels);

            bench.iter(|| {
                collapse::<U6>(&input_graph, output_graph.clone(), None, None)
            })
        }
    }
}

mod graphs {
    use bencher::Bencher;
    use wfc_rust::graph::graph::*;
    use wfc_rust::utils::{hash_map, hash_set};
    use wfc_rust::multiset::{Multiset, MultisetTrait};
    use std::iter::FromIterator;
    use hashbrown::HashMap;
    use rand::prelude::*;
    use nalgebra::U6;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    pub fn graph_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
        ];

        let graph = Graph::<U6>::new(graph_vertices, graph_edges(), Multiset::from_row_slice_u(&[1, 2, 1]));

        bench.iter(|| {
            graph.rules()
        })
    }

    pub fn graph_observe(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[3, 3, 0, 0]),
            Multiset::from_row_slice_u(&[3, 3, 1, 2]),
            Multiset::from_row_slice_u(&[0, 3, 0, 2]),
            Multiset::from_row_slice_u(&[3, 0, 0, 0])
        ];

        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[3, 3, 1, 2]));
        let mut test_rng = StdRng::seed_from_u64(2);
        let index = 1;

        bench.iter(|| {
            graph.clone().observe(&mut test_rng, &index)
        });
    }

    pub fn graph_constrain_true(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 1, 1])
        ];

        let constraint = Multiset::from_row_slice_u(&[1, 1, 0, 0]);
        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[1, 1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }

    pub fn graph_constrain_false(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 0])
        ];

        let constraint = Multiset::from_row_slice_u(&[1, 1, 1]);
        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }
}


benchmark_group!(
    benches,
    collapse::bench_collapse,
    // graphs::graph_rules,
    // graphs::graph_observe,
    // graphs::graph_constrain_true,
    // graphs::graph_constrain_false
);
benchmark_main!(benches);

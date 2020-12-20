#[macro_use]
extern crate bencher;
extern crate nalgebra;

mod collapse {
    use bencher::Bencher;
    use nalgebra::{U4, VectorN};
    use wfc_rust::graph::graph::Graph;
    use wfc_rust::io::text_parser::parse;
    use wfc_rust::io::utils::make_edges_cardinal_grid;
    use wfc_rust::wfc::collapse::collapse;

    type MS4 = VectorN<u32, U4>;

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse::<MS4>("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels;
            let output_vertices: Vec<MS4> = vec![all_labels; out_width * out_depth];
            let output_edges = make_edges_cardinal_grid(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges, all_labels);

            bench.iter(|| collapse(&input_graph, output_graph.clone(), None, None))
        }
    }
}

mod graphs {
    use bencher::Bencher;
    use nalgebra::{U6, VectorN};
    use wfc_rust::graph::graph::*;
    use wfc_rust::multiset::Multiset;
    use wfc_rust::utils::hash_map;

    type MS6 = VectorN<u16, U6>;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ])
    }

    pub fn graph_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<MS6> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
        ];

        let graph = Graph::new(
            graph_vertices,
            graph_edges(),
            Multiset::from_row_slice_u(&[1, 2, 1]),
        );

        bench.iter(|| graph.rules())
    }
}

benchmark_group!(
    benches,
    collapse::bench_collapse,
    // graphs::graph_rules,
);
benchmark_main!(benches);

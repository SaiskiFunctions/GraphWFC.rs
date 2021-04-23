#[macro_use]
extern crate bencher;
extern crate nalgebra;

mod collapse {
    use bencher::Bencher;
    use wfc_rust::graph::graph::Graph;
    use wfc_rust::io::text_parser::parse;
    use wfc_rust::io::misc_utils::make_edges_8_way_grid;
    use wfc_rust::wfc::collapse::collapse;

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse("resources/test/tosashimizu_model.txt", true) {
            let all_labels = input_graph.all_labels;
            let output_vertices = vec![all_labels; out_width * out_depth];
            let output_edges = make_edges_8_way_grid(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges, all_labels);

            bench.iter(|| collapse(&input_graph.rules(), output_graph.clone(), Some(1), None))
        }
    }
}

benchmark_group!(
    benches,
    collapse::bench_collapse,
);
benchmark_main!(benches);

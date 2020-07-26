#[macro_use]
extern crate bencher;

use bencher::Bencher;
use wfc_rust::wfc::collapse::collapse;
use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges};
use wfc_rust::graph::graph::{Labels, Graph};


fn bench_collapse(bench: &mut Bencher) {
    let out_width = 10;
    let out_depth = 1000;

    if let Ok((input_graph, _)) = parse("resources/test/tosashimizu_model.txt") {
        let all_labels = input_graph.all_labels();
        let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
        let output_edges = make_nsew_grid_edges(out_width, out_depth);
        let output_graph = Graph::new(output_vertices, output_edges);

        bench.iter(|| {collapse(&input_graph, output_graph.clone(), Some(134522), None)})
    }
}


benchmark_group!(benches, bench_collapse);
benchmark_main!(benches);

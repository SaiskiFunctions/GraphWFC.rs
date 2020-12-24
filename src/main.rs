use nalgebra::{VectorN, U6};
use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser::{parse, render};
use wfc_rust::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use wfc_rust::multiset::Multiset;
use wfc_rust::wfc::collapse::collapse;

fn run_collapse<S: Multiset>(input: &str, output: &str, width: usize, depth: usize) {
    if let Ok((input_graph, keys)) = parse::<S>(input) {
        let all_labels = input_graph.all_labels.clone();
        let output_vertices = vec![all_labels.clone(); width * depth];
        // let output_edges = make_edges_cardinal_grid(width, depth);
        let output_edges = make_edges_8_way_grid(width, depth);
        let output_graph = Graph::new(output_vertices, output_edges, all_labels);
        let collapsed_graph = collapse(&input_graph, output_graph, Some(134522));
        render(output, &collapsed_graph, &keys, width);
    }
}

// BUG: out_width is divded by 2 when rendered
fn main() {
    let input = "resources/test/tosashimizu_model.txt";
    let output = "resources/test/tosashimizu_output3.txt";
    let out_width = 1000;
    let out_depth = 1000;

    run_collapse::<VectorN<u16, U6>>(input, output, out_width, out_depth);
}

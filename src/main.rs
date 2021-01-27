use nalgebra::{VectorN, U6};
use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser::{parse, render};
use wfc_rust::io::image_olm_parser;
use wfc_rust::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use wfc_rust::multiset::Multiset;
use wfc_rust::wfc::collapse::collapse;

fn run_collapse<S: Multiset>(input: &str, chunk_size: u32, output: &str, width: usize, depth: usize) {
    let (rules, keys, all_labels, chunks) = image_olm_parser::parse::<S>(input, chunk_size);
    let output_edges = make_edges_cardinal_grid(width, depth);
    let output_vertices = vec![all_labels.clone(); width * depth];
    let output_graph = Graph::new(output_vertices, output_edges, all_labels);
    let collapsed_graph = collapse(&rules, output_graph, Some(134522), false);
    image_olm_parser::render(output, &collapsed_graph, &keys, &chunks, width);

    if let Ok((input_graph, keys)) = image_olm_parser::parse::<S>(input, chunk_size) {
        let all_labels = input_graph.all_labels.clone();
        let output_vertices = vec![all_labels.clone(); width * depth];
        // let output_edges = make_edges_cardinal_grid(width, depth);
        let output_edges = make_edges_8_way_grid(width, depth);
        let output_graph = Graph::new(output_vertices, output_edges, all_labels);
        let collapsed_graph = collapse(input_graph.rules(), output_graph, Some(134522), false);
        render(output, &collapsed_graph, &keys, width);
    }
}

// BUG: out_width is divded by 2 when rendered
fn main() {
    let input = "resources/test/tosashimizu_model.txt";
    let output = "resources/test/tosashimizu_output3.txt";
    let out_width = 1000;
    let out_depth = 1000;

    run_collapse::<VectorN<u16, U6>>(input, 2, output, out_width, out_depth);
}

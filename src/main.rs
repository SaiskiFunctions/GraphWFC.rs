use nalgebra::{VectorN, U6, U100, U50, U25, U13, U20};
use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser::{parse, render};
use wfc_rust::io::image_olm_parser;
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
        let collapsed_graph = collapse(&input_graph.rules(), output_graph, Some(134522), false);
        render(output, &collapsed_graph, &keys, width);
    }
}

fn run_olm<S: Multiset>(input: &str, chunk_size: u32, output: &str, width: usize, depth: usize) {
    let (rules, keys, all_labels, chunks) = image_olm_parser::parse::<S>(input, chunk_size);
    let graph_width = (width / chunk_size as usize); // in chunks
    let graph_depth = (depth / chunk_size as usize); // in chunks
    let output_edges = make_edges_8_way_grid(graph_width, graph_depth);
    println!("{:?}", output_edges.get(&0).unwrap());
    // TODO: Validation check for chunk dividing into pixels
    let output_vertices = vec![all_labels.clone();graph_width * graph_depth];
    let output_graph = Graph::new(output_vertices, output_edges, all_labels);
    let collapsed_graph = collapse(&rules, output_graph, Some(134522), false);
    image_olm_parser::render(output, collapsed_graph, &keys, &chunks, (width, depth), chunk_size);
}

// BUG: out_width is divded by 2 when rendered
fn main() {
    let input = "resources/test/Qud.png";
    let output = "resources/test/test_result_6.png";
    let out_width = 70;
    let out_depth = 70;

    run_olm::<VectorN<u16, U100>>(input, 3, output, out_width, out_depth);
}

// fn main() {
//     let input = "resources/test/tosashimizu_model.txt";
//     let output = "resources/test/tosashimizu_output3.txt";
//     let out_width = 20;
//     let out_depth = 20;
//
//     run_collapse::<VectorN<u16, U6>>(input, output, out_width, out_depth);
// }

use wfc_rust::wfc::collapse::collapse;
use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges, render};
use nalgebra::U6;
use wfc_rust::multiset::Multiset;

// BUG: out_width is divded by 2 when rendered
fn main() {
    let out_width = 20;
    let out_depth = 10;

    if let Ok((input_graph, keys)) = parse::<U6>("resources/test/tosashimizu_model.txt") {
        let all_labels = input_graph.all_labels;
        let output_vertices: Vec<Multiset<U6>> = vec![all_labels.clone(); out_width * out_depth];
        let output_edges = make_nsew_grid_edges(out_width, out_depth);
        let output_graph = Graph::<U6>::new(output_vertices, output_edges, all_labels);
        if let Some(collapsed_graph) = collapse(&input_graph, output_graph, Some(134522), None) {
            render("resources/test/tosashimizu_output3.txt", &collapsed_graph, &keys, out_width);
        }
    }
}

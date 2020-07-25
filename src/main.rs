mod graph;
mod io;
mod utils;
mod wfc;

use crate::wfc::collapse::collapse;
use crate::graph::graph::{Graph, Edges, Labels, GraphGen};
use crate::utils::{hash_set, hash_map};
use crate::io::text_parser::{parse, make_nsew_grid_edges, render};

// BUG: out_width is divded by 2 when rendered
fn main() {
    let out_width = 20;
    let out_depth = 10;

    if let Ok((input_graph, keys)) = parse("resources/test/tosashimizu_model.txt") {
        // println!("rules: {:?}", input_graph.rules());
        // println!("Key: {:?}", keys);
        let all_labels = input_graph.all_labels();
        let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
        let output_edges = make_nsew_grid_edges(out_width, out_depth);
        let output_graph = Graph::new(output_vertices, output_edges);
        if let Some(collapsed_graph) = collapse(input_graph, output_graph, Some(134522), None) {
            render("resources/test/tosashimizu_output.txt", &collapsed_graph, &keys, out_width);
        }
    }
}

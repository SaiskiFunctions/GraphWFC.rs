mod graph;
mod io;
mod utils;
mod wfc;

use crate::wfc::collapse::collapse;
use crate::graph::graph::{Graph, Edges, Labels};
use crate::utils::{hash_set, hash_map};
use crate::io::text_parser::{parse, make_nsew_grid_edges, render};


fn main() {
    let out_width = 20;
    let out_depth = 20;

    if let Ok((input_graph, keys)) = parse("resources/test/another_emoji.txt") {
        println!("rules: {:?}", input_graph.rules());
        println!("Key: {:?}", keys);
        let all_labels = input_graph.all_labels();
        let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
        let output_edges = make_nsew_grid_edges(out_width, out_depth);
        let output_graph = Graph::new(output_vertices, output_edges);
        if let Some(collapsed_graph) = collapse(input_graph, output_graph, Some(134522), None) {
            render("resources/test/render_emoji5.txt", &collapsed_graph, &keys, out_width);
        }
    }
}

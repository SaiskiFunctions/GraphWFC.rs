use std::collections::{HashSet, HashMap};
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;
use crate::graph::graph::{Graph, Labels, Edges, EdgeDirection, VertexIndex, VertexLabel};
use crate::utils::hash_set;

pub fn parse(filename: &str) -> Result<(Graph, HashMap<usize, char>), Error> {
    read_to_string(filename).map(|string| {
        let lines: Vec<&str> = string.split('\n').filter(|l| !l.is_empty()).collect();
        let num_lines = lines.len();
        let line_len = lines.index(0).chars().count();

        let mut edges: Edges = HashMap::new();
        let mut key_set: HashSet<char> = HashSet::new();

        lines.iter().enumerate().for_each(|(line_index, line)| {
            line.chars().enumerate().for_each(|(character_index, character)| {
                key_set.insert(character);

                let mut direction_pairs = Vec::new();
                //NORTH = 0
                if line_index > 0 {
                    direction_pairs.push(((line_index - 1) * line_len + character_index, 0))
                }
                //SOUTH = 1
                if line_index < num_lines - 1 {
                    direction_pairs.push(((line_index + 1) * line_len + character_index, 1))
                }
                //WEST = 3
                if character_index > 0 {
                    direction_pairs.push(((character_index - 1) + line_index * line_len, 3))
                }
                //EAST = 2
                if character_index < line_len - 1 {
                    direction_pairs.push(((character_index + 1) + line_index * line_len, 2))
                }

                let direction_pairs = direction_pairs.into_iter().map(|(i, d)| {
                    (i as VertexIndex, d as EdgeDirection)
                }).collect::<Vec<(VertexIndex, EdgeDirection)>>();

                let this_vertex_index = ((line_index * line_len) + character_index) as VertexIndex;
                edges.insert(this_vertex_index, direction_pairs);
            });
        });

        let key_vec: Vec<_> = key_set.into_iter().collect();

        let vertices: Vec<Labels> = string.chars().filter(|c| c != &'\n').map(|c| {
            let label = key_vec.iter().position(|&e| e == c).unwrap();
            hash_set(&[label as i32])
        }).collect();

        let keys: HashMap<_, _> = key_vec.into_iter().enumerate().collect();

        (Graph::new(vertices, edges), keys)
    })
}

pub fn make_nsew_grid_edges(width: usize, depth: usize) -> Edges {
    let mut edges: Edges = HashMap::new();
    for depth_index in 0..depth {
        for width_index in 0..width {

            let mut direction_pairs = Vec::new();
            //NORTH = 0
            if depth_index > 0 {
                direction_pairs.push(((depth_index - 1) * width + width_index, 0))
            }
            //SOUTH = 1
            if depth_index < depth - 1 {
                direction_pairs.push(((depth_index + 1) * width + width_index, 1))
            }
            //WEST = 3
            if width_index > 0 {
                direction_pairs.push(((width_index - 1) + depth_index * width, 3))
            }
            //EAST = 2
            if width_index < width - 1 {
                direction_pairs.push(((width_index + 1) + depth_index * width, 2))
            }

            let direction_pairs = direction_pairs.into_iter().map(|(i, d)| {
                (i as VertexIndex, d as EdgeDirection)
            }).collect::<Vec<(VertexIndex, EdgeDirection)>>();

            let this_vertex_index = ((depth_index * width) + width_index) as VertexIndex;
            edges.insert(this_vertex_index, direction_pairs);
        };
    };
    edges
}

pub fn render(filename: &str, graph: &Graph, key: &HashMap<usize, char>, width: usize) {
    // println!("GRAPH: {:?}", graph);
    let rendered_vertices: Vec<char> = graph.vertices.iter().map(|labels| {
        let k = *labels.iter().next().unwrap() as usize;
        *key.index(&k)
    }).collect();
    let lines: String = rendered_vertices.chunks_exact(width).map(|chunk| {
        chunk.iter().collect::<String>() + "\n"
    }).collect::<String>();
    if let Ok(_) = write(filename, lines) {};
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse("resources/test/easy_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse("resources/test/medium_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}

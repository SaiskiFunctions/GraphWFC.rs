use std::fs::read_to_string;
use std::io::Error;
use std::collections::{HashSet, HashMap};
use crate::wfc::graph::{Graph, Labels, Edges, EdgeDirection, VertexIndex, VertexLabel};
use crate::utils::{hash_set};

// 1. Load file into string
// 2. Parse string into graph
/*
    Grid with 4 directions
    \n -> vertical spacing

    aaaa   \n
    aaaaaaa\n
    aaaa   \n

    Vec<Vec<Char>>


    HashMap<Chars, Label>

    input.string("\n")


*/

pub fn parse(filename: &str) -> Result<(Graph, HashMap<usize, char>), Error> {
    read_to_string(filename).map(|string| {
        let lines: Vec<&str> = string.split('\n').collect();

        let mut edges: Edges = HashMap::new();
        let mut key_set: HashSet<char> = HashSet::new();

        lines.iter().enumerate().for_each(|(line_index, line)| {
            line.chars().enumerate().for_each(|(character_index, character)| {
                key_set.insert(character);

                let this_vertex_index: VertexIndex = ((line_index + 1) * character_index) as VertexIndex;
                let mut direction_pairs = Vec::new();
                //NORTH = 0
                if lines.get(line_index - 1).is_some() {
                    direction_pairs.push(((line_index - 1) * character_index, 0))
                }

                //SOUTH = 1
                if lines.get(line_index + 1).is_some() {
                    direction_pairs.push(((line_index + 1) * character_index, 1))
                }

                //EAST
                if line.get((character_index + 1)..character_index).is_some() {
                    direction_pairs.push(((character_index + 1) * (line_index + 1), 2))
                }

                //WEST
                if line.get((character_index - 1)..character_index).is_some() {
                    direction_pairs.push(((character_index - 1) * (line_index + 1), 3))
                }

                let direction_pairs: Vec<(VertexIndex, EdgeDirection)> = direction_pairs.iter().map(|(i, d)| {
                    (*i as VertexIndex, *d as EdgeDirection)
                }).collect();

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        if let Ok((graph, keys)) = parse("resources/test/emoji.txt") {
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }
}

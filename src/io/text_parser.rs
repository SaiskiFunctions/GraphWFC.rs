use std::fs::read_to_string;
use std::io::Error;
use std::collections::{HashSet, HashMap};
use crate::wfc::graph::{Graph, Labels, Edges, EdgeDirection, VertexIndex, VertexLabel};

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

pub fn parse(filename: &str) -> Result<Graph, Error> {
    read_to_string(filename).map(|string| {
        let lines: Vec<&str> = string.split('\n').collect();

        let mut vertices: Vec<Labels> = Vec::new();
        let mut edges: Edges = HashMap::new();
        let mut key: HashSet<char> = HashSet::new();

        lines.iter().enumerate().for_each(|(line_index, line)| {
            line.chars().enumerate().for_each(|(character_index, character)| {
                key.insert(character);

                // let line_index = line_index as i32;
                // let character_index = character_index as i32;

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

        /*
        lines.flatten()
        key.to_vec();
        */

        Graph::empty()
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read() {
        match parse("resources/test/emoji.txt") {
            Ok(string) => println!("{}", string),
            Err(e) => println!("hi {}", e)
        };
    }
}

use crate::graph::graph::{Edges, Graph};
use crate::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use hashbrown::HashMap;
use num_traits::One;
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;
use bimap::BiHashMap;
use crate::MSu16xNU;
use itertools::Itertools;

pub type CharKeyBimap = BiHashMap<usize, char>;

pub fn parse(filename: &str) -> Result<(Graph, CharKeyBimap), Error> {
    read_to_string(filename).map(|string| {
        let edges = make_edges(&string);
        let char_frequency = char_frequency(&string);
        let char_keys: CharKeyBimap = char_keys(&char_frequency);
        let all_labels = make_all_labels(&char_frequency, &char_keys);
        let vertices = make_vertices(&string, &char_frequency, &char_keys, true);

        (Graph::new(vertices, edges, all_labels), char_keys)
    })
}

// 3 syllabale rhyming couplet poem to remind us why we should use sorted data structures
// Mountain tree
// Tori sea
// Hashmap lame
// Vecs have game
fn char_keys(char_frequency: &HashMap<char, u16>) -> CharKeyBimap {
    char_frequency
        .keys()
        .copied()
        .sorted_by(|x, y| x.cmp(y))
        .enumerate()
        .collect()
}

fn char_frequency(string: &str) -> HashMap<char, u16> {
    string
        .chars()
        .filter(|char| char != &'\n')
        .fold(HashMap::new(), |mut map, char| {
            map
                .entry(char)
                .and_modify(|freq| *freq += 1)
                .or_insert(1);
            map
        })
}

fn make_all_labels(
    char_frequency: &HashMap<char, u16>,
    char_keys: &CharKeyBimap,
) -> MSu16xNU {
    (0..char_keys.len()).map(|index| {
        let character = char_keys.get_by_left(&index).unwrap();
        *char_frequency.index(character)
    }).collect()
}

fn make_vertices(
    string: &str,
    char_frequency: &HashMap<char, u16>,
    char_keys: &CharKeyBimap,
    directional: bool,
) -> Vec<MSu16xNU> {
    string
        .chars()
        .filter(|c| c != &'\n')
        .map(|c| {
            let index = *char_keys.get_by_right(&c).unwrap();
            let mut set = MSu16xNU::empty();
            if directional {
                set.insert(index, 1);
            } else {
                set.insert(index, *char_frequency.index(&c));
            }
            set
        })
        .collect()
}

fn make_edges(string: &str) -> Edges {
    let width = string
        .chars()
        .enumerate()
        .find_map(|(index, char)| if char == '\n' { Some(index) } else { None })
        .unwrap_or_else(|| string.chars().count());
    let height = string.split('\n').filter(|l| !l.is_empty()).count();
    // make_edges_cardinal_grid(width, height)
    make_edges_8_way_grid(width, height)
}

const CROSS: char = '❌';

pub fn render(
    filename: &str,
    graph: &Graph,
    key: &CharKeyBimap,
    width: usize,
) {
    let rendered_vertices: Vec<char> = graph
        .vertices
        .iter()
        .map(|labels| {
            labels
                .is_singleton()
                .then(|| key.get_by_left(&labels.imax()).copied())
                .flatten()
                .unwrap_or(CROSS)
        })
        .collect();
    let lines: String = rendered_vertices
        .chunks_exact(width)
        .map(|chunk| chunk.iter().collect::<String>() + "\n")
        .collect::<String>();
    if write(filename, lines).is_ok() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{VectorN, U6, U7};

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse("resources/test/easy_emoji.txt") {
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse("resources/test/medium_emoji.txt") {
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}

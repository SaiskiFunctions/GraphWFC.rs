use crate::graph::graph::{Edges, Graph, Vertices};
use crate::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use crate::MSu16xNU;
use std::fs::{read_to_string, write};
use std::io::Error;
use std::iter;
use indexmap::IndexMap;

pub fn parse(filename: &str, intercardinals: bool) -> Result<(Graph, IndexMap<char, u16>), Error> {
    read_to_string(filename).map(|string| {
        let edges = make_edges(&string, intercardinals);
        let (char_frequency, vertices) = char_maps(&string);
        let all_labels = char_frequency.values().collect();
        (Graph::new(vertices, edges, all_labels), char_frequency)
    })
}

// 3 syllabale rhyming couplet poem to remind us why we should use sorted data structures
// Mountain tree
// Tori sea
// Hashmap lame
// Vecs have game
fn char_maps(string: &str) -> (IndexMap<char, u16>, Vec<MSu16xNU>) {
    let mut char_frequencies: IndexMap<char, u16> = IndexMap::new();
    let vertices: Vec<MSu16xNU> = string
        .chars()
        .filter(|char| char != &'\n')
        .map(|char| {
            let label = char_frequencies.entry(char).index();
            char_frequencies.entry(char).and_modify(|f| *f += 1).or_insert(1);
            let mut set = MSu16xNU::empty();
            set.insert(label, 1);
            set
        })
        .collect();
    (char_frequencies, vertices)
}

fn make_edges(string: &str, intercardinals: bool) -> Edges {
    let width = string.chars().position(|char| char == '\n').unwrap();
    let height = string.split('\n').filter(|l| !l.is_empty()).count();
    if intercardinals {
        make_edges_8_way_grid(width, height)
    } else {
        make_edges_cardinal_grid(width, height)
    }
}

const CONTRADICT_CHAR: char = '❌';

pub fn render(
    filename: &str,
    graphs: Vec<Vertices>,
    key: &IndexMap<char, u16>,
    width: usize,
) {
    let graph = graphs.last().unwrap(); // text parser does do support progress renders yet
    let lines: String = graph
        .chunks_exact(width)
        .map(|chunk| {
            chunk
                .iter()
                .map(|labels| {
                    labels
                        .is_singleton()
                        .then(|| key.get_index(labels.imax()).map(|t| *t.0))
                        .flatten()
                        .unwrap_or(CONTRADICT_CHAR)
                })
                .chain(iter::once('\n'))
        })
        .flatten()
        .collect::<String>();
    if write(filename, lines).is_ok() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse("resources/test/easy_emoji.txt", true) {
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse("resources/test/medium_emoji.txt", true) {
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}

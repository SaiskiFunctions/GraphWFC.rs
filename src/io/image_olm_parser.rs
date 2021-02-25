use crate::graph::graph::{Rules, Edges, Graph};
use crate::io::{
    limit_iter::Limit,
    sub_matrix::SubMatrix,
    tri_wave::TriWave,
    utils::Rotation,
};
use crate::multiset::Multiset;
use crate::utils::{index_to_coords, is_inside, coords_to_index};
use crate::wfc::collapse;

use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, Rgb, RgbImage};
use itertools::Itertools;
use nalgebra::DMatrix;
use num_traits::One;
use std::collections::HashSet;
use std::ops::{IndexMut, Index};
use std::convert::TryFrom;

const GREEN: image::Rgb<u8> = image::Rgb([0, 255, 0]);

type Chunk = DMatrix<usize>;
type PixelKeys = BiMap<usize, Rgb<u8>>;

// jump by chunk and render the pixels inside each chunk
//    0    2    3    <- chunk_coords x component
//   ┏━━━┓┏━━━┓┏━━━┓     0 1 <- pixel_coords x compoent
// 0 ┃▆ ▆┃┃   ┃┃   ┃   0 ▆ ▆
//   ┃▆ ▆┃┃   ┃┃   ┃   1 ▆ ▆
//   ┗━━━┛┗━━━┛┗━━━┛   ^ pixel-coords y component
//   ┏━━━┓┏━━━┓┏━━━┓
// 1 ┃   ┃┃   ┃┃   ┃
//   ┃   ┃┃   ┃┃   ┃
//   ┗━━━┛┗━━━┛┗━━━┛
// ^ chunk_coords y component

pub fn render<S: Multiset>(
    filename: &str,
    graph: Graph<S>,
    key: &PixelKeys,
    chunks: &[Chunk],
    (width, height): (usize, usize),
    chunk_size: usize,
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks
    let contradiction_key = key.len();

    graph
        .vertices
        .into_iter()
        .map(|vertex| vertex.get_non_zero().map(|i| chunks.index(i).clone()))
        .enumerate()
        .for_each(|(index, opt_chunk)| {
            let (x, y) = index_to_coords(index, graph_width);
            let top_left_pix_x = x * chunk_size;
            let top_left_pix_y = y * chunk_size;

            let chunk = match opt_chunk {
                None => DMatrix::from_element(chunk_size, chunk_size, contradiction_key),
                Some(chunk) => chunk
            };

            chunk
                .iter()
                .enumerate()
                .for_each(|(pixel_index, pixel_alias)| {
                    let (pixel_y, pixel_x) = index_to_coords(pixel_index, chunk_size);
                    let pixel = output_image.get_pixel_mut((top_left_pix_x + pixel_x) as u32, (top_left_pix_y + pixel_y) as u32);
                    *pixel = *key.get_by_left(pixel_alias).unwrap_or(&GREEN);
                });
        });

    output_image.save(filename).unwrap();
}

// TODO: handle unwrap of image::open properly
pub fn parse<S: Multiset>(filename: &str, chunk_size: usize) -> (Rules<S>, PixelKeys, S, Vec<Chunk>) {
    let img = image::open(filename).unwrap().to_rgb8();
    let pixel_aliases = alias_pixels(&img);
    let chunks = chunk_image(img, chunk_size, &pixel_aliases, false);
    let overlap_rules = overlaps::<S>(&chunks, chunk_size);

    let mut all_labels = S::empty(chunks.len());
    for i in 0..chunks.len() {
        *all_labels.index_mut(i) = One::one()
    }

    let mut pruned_rules: Rules<S> = HashMap::new();

    (0..all_labels.count_non_zero())
        .for_each(|label| {
            let pruned_graph = propagate_overlaps(&all_labels, &overlap_rules, chunk_size, label);

            real_vertex_indexes(chunk_size)
                .into_iter()
                .enumerate()
                .for_each(|(direction, index)| {
                    let set = pruned_graph.vertices.index(index);
                    if !set.is_empty_m() {
                        pruned_rules.insert((direction as u16, label), set.clone());
                    }
                });
        });

    (pruned_rules, pixel_aliases, all_labels, chunks)
}

fn real_vertex_indexes(chunk_size: usize) -> Vec<usize> {
    let dim = (3 * chunk_size) - (chunk_size - 1);
    let step = chunk_size - 1;
    vec![
        0,                                      // NW
        step + 1,                               // N
        (step + 1) * 2,                         // NE
        dim * chunk_size,                       // W
        // dim * chunk_size + step + 1
        dim * chunk_size + (step + 1) * 2,      // E
        dim * chunk_size * 2,                   // SW
        dim * chunk_size * 2 + step + 1,        // S
        dim * chunk_size * 2 + (step + 1) * 2,  // SE
    ]
}

fn sub_images(image: RgbImage, chunk_size: usize) -> impl Iterator<Item=RgbImage> {
    let chunk_size_32: u32 = TryFrom::try_from(chunk_size)
        .expect("chunk_size too large, cannot convert to u32");

    let height_iter = 0..(image.dimensions().1) - (chunk_size_32 - 1);
    let width_iter = 0..(image.dimensions().0) - (chunk_size_32 - 1);

    height_iter
        .cartesian_product(width_iter)
        .map(move |(y, x)| {
            imageops::crop_imm(&image, x, y, chunk_size_32, chunk_size_32).to_image()
        })
}

fn alias_sub_image(image: RgbImage, pixel_aliases: &PixelKeys) -> Vec<usize> {
    image
        .pixels()
        .map(|p| *pixel_aliases.get_by_right(&p).unwrap())
        .collect()
}

fn alias_pixels(image: &RgbImage) -> PixelKeys {
    image
        .pixels()
        .fold(HashSet::<Rgb<u8>>::new(), |mut acc, pixel| {
            acc.insert(*pixel);
            acc
        })
        .into_iter()
        .enumerate()
        .collect()
}

fn chunk_image(
    image: RgbImage,
    chunk_size: usize,
    pixel_aliases: &PixelKeys,
    rotate: bool,
) -> Vec<Chunk> {
    sub_images(image, chunk_size)
        .map(|sub_image| alias_sub_image(sub_image, pixel_aliases))
        .fold(HashSet::new(), |mut acc, pixels| {
            let chunk = DMatrix::from_row_slice(chunk_size, chunk_size, &pixels);

            acc.insert(chunk.clone());

            if rotate {
                let chunk_r90 = chunk.rotate_90();
                acc.insert(chunk_r90.clone());

                let chunk_r180 = chunk_r90.rotate_90();
                acc.insert(chunk_r180.clone());

                let chunk_r270 = chunk_r180.rotate_90();
                acc.insert(chunk_r270);
            }
            acc
        }).into_iter().collect()
}

type Position = (usize, usize);
type Size = (usize, usize);
type Direction = u16;

fn sub_chunk_positions(chunk_size: usize) -> Vec<(Position, Size, Direction)> {
    let period = chunk_size * 2 - 1;
    let positions = Limit::new(chunk_size).zip(TriWave::new(chunk_size)).take(period);
    let pos_cart_prod = positions.clone().cartesian_product(positions);

    pos_cart_prod
        .map(|((y_position, y_size), (x_position, x_size))| (
            (x_position, y_position),
            (x_size + 1, y_size + 1)
        ))
        .filter(|(_, (width, height))| width != &(chunk_size) || height != &(chunk_size))
        .enumerate()
        .map(|(direction, (position, size))| (
            position,
            size,
            direction as u16
        ))
        .collect()
}

fn overlaps<S: Multiset>(chunks: &[Chunk], chunk_size: usize) -> Rules<S> {
    chunks
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (index, chunk)| {
            let sub_positions = sub_chunk_positions(chunk_size);
            sub_positions
                .iter()
                .for_each(|(position, size, direction)| {
                    let sub_chunk = chunk.sub_matrix(*position, *size);
                    let reverse_index = sub_positions.len() - 1 - *direction as usize;
                    let (rev_pos, rev_size, _) = sub_positions[reverse_index];
                    chunks
                        .iter()
                        .enumerate()
                        .for_each(|(other_index, other_chunk)| {
                            // find mirrored sub chunk
                            let other_sub_chunk = other_chunk.sub_matrix(rev_pos, rev_size);
                            if sub_chunk == other_sub_chunk {
                                acc
                                    .entry((*direction, index))
                                    .and_modify(|labels| *labels.index_mut(other_index) = One::one())
                                    .or_insert({
                                        let mut set = S::empty(chunks.len());
                                        *set.index_mut(other_index) = One::one();
                                        set
                                    });
                            }
                        })
                });
            acc
        })
}

// Create a raw graph for pruning
fn create_raw_graph<S: Multiset>(all_labels: &S, chunk_size: usize, (height, width): (usize, usize)) -> Graph<S> {
    // pixel based graph dimensions
    let v_dim_x = (width * chunk_size) - (chunk_size - 1);
    let v_dim_y = (height * chunk_size) - (chunk_size - 1);

    let vertices: Vec<S> = vec![all_labels.clone(); v_dim_x * v_dim_y];
    // create negative indexed range to offset vertex centered directional field by N
    let signed_chunk_size: i32 = TryFrom::try_from(chunk_size)
        .expect("Cannot convert chunk_size to i32");
    let range = 1 - signed_chunk_size..signed_chunk_size;

    // calculate real cartesian space offest coordinates
    let range_cart_prod = range.clone()
        .cartesian_product(range)
        .filter(|i| i != &(0, 0)); // remove 0 offset for correct directional mapping

    let edges: Edges = vertices
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (index, _)| {
            let (x, y) = index_to_coords(index, v_dim_x);
            range_cart_prod.clone()
                .map(|(y_offset, x_offset)| (y as i32 + y_offset, x as i32 + x_offset))
                .enumerate()
                // remove coordinates outside of graph
                .filter(|(_, offsets)| is_inside(*offsets, (v_dim_x, v_dim_y)))
                .for_each(|(direction, (y_offset, x_offset))| {
                    let other_index = coords_to_index(x_offset as usize, y_offset as usize, v_dim_x);
                    acc
                        .entry(index as u32)
                        .and_modify(|v| v.push((other_index as u32, direction as u16)))
                        .or_insert(vec![(other_index as u32, direction as u16)]);
                });
            acc
        });

    Graph::new(vertices, edges, all_labels.clone())
}

fn propagate_overlaps<S: Multiset>(all_labels: &S, rules: &Rules<S>, chunk_size: usize, label: usize) -> Graph<S> {
    let mut raw_graph = create_raw_graph(all_labels, chunk_size, (3, 3));
    let central_vertex = (raw_graph.vertices.len() - 1) / 2;
    raw_graph.vertices.index_mut(central_vertex).determine(label);
    collapse::collapse(rules, raw_graph, None, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::hash_map;
    use image::ImageBuffer;
    use nalgebra::{Vector4, Vector2, Vector1};
    use std::ops::Index;

    #[test]
    fn test_alias_pixels() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);
        assert_eq!(pixel_aliases.len(), 4);
    }

    #[test]
    fn test_chunk_image() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);

        let chunk_vec = chunk_image(img, 2, &pixel_aliases, true);

        let chunk = chunk_vec.index(0).clone();

        let result: HashSet<Chunk> = vec![
            chunk.clone(),
            chunk.rotate_90().rotate_90(),
            chunk.rotate_90(),
            chunk.rotate_90().rotate_90().rotate_90()
        ].into_iter().collect();

        assert_eq!(chunk_vec.len(), 4);
        assert_eq!(chunk_vec.into_iter().collect::<HashSet<Chunk>>(), result);
    }

    #[test]
    fn test_subchunk_positions() {
        let sub_chunks = vec![
            ((0, 0), (1, 1), 0),
            ((0, 0), (2, 1), 1),
            ((1, 0), (1, 1), 2),
            ((0, 0), (1, 2), 3),
            // ((0, 0), (2, 2), 4) --> Implicit full overlap removed
            ((1, 0), (1, 2), 4),
            ((0, 1), (1, 1), 5),
            ((0, 1), (2, 1), 6),
            ((1, 1), (1, 1), 7)
        ];
        assert_eq!(sub_chunk_positions(2), sub_chunks);
    }

    #[test]
    fn test_overlaps() {
        let chunks_n2 = vec![
            DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]),
            DMatrix::from_row_slice(2, 2, &vec![3, 2, 0, 1]),
            DMatrix::from_row_slice(2, 2, &vec![2, 0, 3, 1])
        ];

        let mut overlaps_n2: Rules<Vector4<u32>> = HashMap::new();
        overlaps_n2.insert((5, 0), Multiset::from_row_slice_u(&[0, 1, 0, 0]));
        overlaps_n2.insert((0, 1), Multiset::from_row_slice_u(&[1, 0, 0, 0]));
        overlaps_n2.insert((6, 1), Multiset::from_row_slice_u(&[1, 0, 0, 0]));
        overlaps_n2.insert((1, 0), Multiset::from_row_slice_u(&[0, 1, 0, 0]));
        overlaps_n2.insert((2, 1), Multiset::from_row_slice_u(&[1, 0, 0, 0]));
        overlaps_n2.insert((7, 0), Multiset::from_row_slice_u(&[0, 1, 0, 0]));
        overlaps_n2.insert((2, 2), Multiset::from_row_slice_u(&[0, 1, 0, 0]));
        overlaps_n2.insert((5, 1), Multiset::from_row_slice_u(&[0, 0, 1, 0]));

        let result_n2 = overlaps(&chunks_n2, 2);
        assert_eq!(result_n2, overlaps_n2);

        let chunks_n3 = vec![
            DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
            DMatrix::from_row_slice(3, 3, &vec![9, 10, 11, 12, 13, 14, 15, 16, 0])
        ];

        let mut overlaps_n3: Rules<Vector2<u32>> = HashMap::new();
        overlaps_n3.insert((0, 0), Multiset::from_row_slice_u(&[0, 1]));
        overlaps_n3.insert((23, 1), Multiset::from_row_slice_u(&[1, 0]));

        let result_n3 = overlaps(&chunks_n3, 3);

        assert_eq!(result_n3, overlaps_n3);

        let chunks_n4 = vec![
            DMatrix::from_row_slice(4, 4, &vec![0, 0, 2, 3,
                                                0, 1, 4, 5,
                                                6, 7, 0, 0,
                                                8, 9, 0, 1])
        ];

        // test overlapping with self only
        let mut overlaps_n4: Rules<Vector2<u32>> = HashMap::new();
        overlaps_n4.insert((8, 0), Multiset::from_row_slice_u(&[1, 0]));
        overlaps_n4.insert((39, 0), Multiset::from_row_slice_u(&[1, 0]));

        let results_n4 = overlaps(&chunks_n4, 4);

        assert_eq!(results_n4, overlaps_n4);
    }

    #[test]
    fn test_create_raw_graph() {
        let chunks_n3 = vec![
            DMatrix::from_row_slice(1, 1, &vec![0])
        ];

        let edges_n3: Edges = hash_map(&[
            (0, vec![(1, 12), (2, 13), (4, 16), (5, 17), (6, 18), (8, 21), (9, 22), (10, 23)]),
            (1, vec![(0, 11), (2, 12), (3, 13), (4, 15), (5, 16), (6, 17), (7, 18), (8, 20), (9, 21), (10, 22), (11, 23)]),
            (2, vec![(0, 10), (1, 11), (3, 12), (4, 14), (5, 15), (6, 16), (7, 17), (8, 19), (9, 20), (10, 21), (11, 22)]),
            (3, vec![(1, 10), (2, 11), (5, 14), (6, 15), (7, 16), (9, 19), (10, 20), (11, 21)]),
            (4, vec![(0, 7), (1, 8), (2, 9), (5, 12), (6, 13), (8, 16), (9, 17), (10, 18), (12, 21), (13, 22), (14, 23)]),
        ]);

        let mut all_labels = Vector1::<u32>::empty(chunks_n3.len());
        for i in 0..chunks_n3.len() {
            *all_labels.index_mut(i) = One::one()
        }

        let raw_graph = create_raw_graph::<Vector1<u32>>(&all_labels, 3, (2, 2));

        assert_eq!(raw_graph.edges.get(&0).unwrap(), edges_n3.get(&0).unwrap());
        assert_eq!(raw_graph.edges.get(&1).unwrap(), edges_n3.get(&1).unwrap());
        assert_eq!(raw_graph.edges.get(&2).unwrap(), edges_n3.get(&2).unwrap());
        assert_eq!(raw_graph.edges.get(&3).unwrap(), edges_n3.get(&3).unwrap());
        assert_eq!(raw_graph.edges.get(&4).unwrap(), edges_n3.get(&4).unwrap());
    }
}

use crate::graph::graph::{Rules, Edges, Graph};
use crate::io::{
    limit_iter::Limit,
    sub_matrix::SubMatrix,
    tri_wave::TriWave,
    utils::Rotation,
};
use crate::utils::{index_to_coords, is_inside, coords_to_index};
use crate::wfc::collapse;

use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, Rgb, RgbImage, Pixel};
use itertools::Itertools;
use nalgebra::{DMatrix, U4};
use std::collections::HashSet;
use std::ops::{IndexMut, Index};
use std::convert::TryFrom;
use linked_hash_map::LinkedHashMap;
use std::ops::{Not};

use crate::MSu16xNU;

const GREEN: image::Rgb<u8> = image::Rgb([0, 255, 0]);

type Chunk = DMatrix<usize>;
type PixelKeys = BiMap<usize, Rgb<u8>>;

pub fn render(
    filename: &str,
    graph: Graph,
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
        // Vec<Multiset> => Vec<Vec<DMatrix>>
        // map each non-zero multiset label into a corresponding DMatrix
        .map(|vertex| {
            vertex
                .into_iter()
                .enumerate()
                .fold(Vec::new(), |mut acc, (label, frequency)| {
                    if frequency > 0 {
                        acc.push(chunks.index(label).clone())
                    }
                    acc
                })
        })
        // Vec<Vec<DMatrix>> => Vec<Option<DMatrix>>
        // map to options depending on whether the set of chunks is empty
        .map(|label_list| label_list.is_empty().not().then(|| label_list))
        .enumerate()
        // for each chunk of the image which may contain multiple overlayed opt_chunks
        .for_each(|(chunk_index, opt_chunks)| {
            let (chunk_x, chunk_y) = index_to_coords(chunk_index, graph_width);
            // project to pixel coordinates
            let top_left_pix_x = chunk_x * chunk_size;
            let top_left_pix_y = chunk_y * chunk_size;

            // Vec<Option<DMatrix>> => Vec<DMatrix>
            let chunks: Vec<DMatrix<usize>> = match opt_chunks {
                // change an empty set of chunks into a chunk of contradiction pixels
                None => vec![DMatrix::from_element(chunk_size, chunk_size, contradiction_key)],
                // otherwise returns chunks
                Some(chunks) => chunks
            };

            let blend_coefficient = 1.0 / chunks.len() as f64;

            let blended_pixel_chunks: Vec<Vec<Rgb<u8>>> = chunks
                // a vec of vecs of pixel aliases
                .iter()
                // map to a vec of vecs of Rgb<u8> pixels
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|pixel_alias| {
                            key
                                .get_by_left(pixel_alias)
                                .copied()
                                .unwrap_or(GREEN)
                                .map(|channel| (channel as f64 * blend_coefficient) as u8)
                        })
                        .collect::<Vec<Rgb<u8>>>()
                }).collect();

            let mut output_pixels: Vec<Rgb<u8>> = vec![Rgb::from([0, 0, 0]); chunk_size * chunk_size];

            blended_pixel_chunks
                .into_iter()
                .for_each(|chunk| {
                    output_pixels
                        .iter_mut()
                        .zip(chunk.iter())
                        .for_each(|(output_pixel, chunk_pixel)| {
                            output_pixel
                                .channels_mut()
                                .iter_mut()
                                .zip(chunk_pixel.channels())
                                .for_each(|(o_pix, b_pix)| {
                                    *o_pix += b_pix
                                })
                        })
                });

            output_pixels
                .into_iter()
                .enumerate()
                .for_each(|(pixel_index, pixel)| {
                    let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
                    let pixel_y = (top_left_pix_y + p_y) as u32;
                    let pixel_x = (top_left_pix_x + p_x) as u32;
                    output_image.put_pixel(pixel_x, pixel_y, pixel);
                });
        });

    output_image.save(filename).unwrap();
}

fn make_empty_pixel() -> Rgb<u8> {
    Rgb::from([0, 0, 0])
}

// TODO: handle unwrap of image::open properly
pub fn parse(filename: &str, chunk_size: usize) -> (Rules, PixelKeys, MSu16xNU, Vec<Chunk>) {
    let img = image::open(filename).unwrap().to_rgb8();
    let pixel_aliases = alias_pixels(&img);
    let chunk_frequencies = chunk_image(img, chunk_size, &pixel_aliases, false);
    // convert frequencies into a list of unique chunks
    // TODO: could use chunk keys directly in future
    let mut chunks: Vec<Chunk> = chunk_frequencies
        .keys()
        .fold(Vec::new(), |mut acc, chunk| {
            acc.push(chunk.clone());
            acc
        });
    let overlap_rules = overlaps(&chunks, chunk_size);

    if chunks.len() > MSu16xNU::len() {
        println!("Chunks LEN: {}", chunks.len());
        panic!("labels multiset not large enough to store all unique chunks")
    }

    // put frequencies into multi set
    let mut all_labels = chunks
        .iter()
        .enumerate()
        .fold(MSu16xNU::empty(), |mut acc, (index, chunk)| {
            let frequency = chunk_frequencies.get(chunk).unwrap();
            acc.insert(index, *frequency as u16);
            // acc.insert(index, 1); // previous implementation
            acc
        });

    let raw_graph = create_raw_graph(&all_labels, chunk_size, (3, 3));
    let mut pruned_rules: Rules = HashMap::new();

    (0..all_labels.count_non_zero())
        .for_each(|label| {
            let pruned_graph = propagate_overlaps(raw_graph.clone(), &overlap_rules, label as usize);

            real_vertex_indexes(chunk_size)
                .iter()
                .enumerate()
                .for_each(|(direction, index)| {
                    let set = pruned_graph.vertices.index(*index);
                    if !set.is_empty() {
                        pruned_rules.insert((direction as u16, label as usize), *set);
                    }
                });
        });

    (pruned_rules, pixel_aliases, all_labels, chunks)
}

const fn real_vertex_indexes(chunk_size: usize) -> [usize; 8] {
    let dim = (3 * chunk_size) - (chunk_size - 1);
    let step = chunk_size - 1;
    [
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

// returns the input image in unique chunks and frequencies of those chunks
fn chunk_image(
    image: RgbImage,
    chunk_size: usize,
    pixel_aliases: &PixelKeys,
    rotate: bool,
) -> LinkedHashMap<Chunk, usize> {
    sub_images(image, chunk_size)
        .map(|sub_image| alias_sub_image(sub_image, pixel_aliases))
        .fold(LinkedHashMap::new(), |mut acc, pixels| {
            let mut chunks: Vec<Chunk> = Vec::new();
            let mut chunk = DMatrix::from_row_slice(chunk_size, chunk_size, &pixels);
            chunks.push(chunk.clone());

            if rotate {
                // rotate through 90° three times using the previous chunk as a starting point
                (0..3)
                    .for_each(|i| {
                        chunk = chunk.rotate_90();
                        chunks.push(chunk.clone());
                    });
            }

            // add each of the new chunks into the frequency>chunk map
            chunks
                .into_iter()
                .for_each(|chunk| {
                    *acc.entry(chunk).or_insert(0) += 1
                });
            acc
        })
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
        .filter(|(_, (width, height))| width != &chunk_size || height != &chunk_size)
        .enumerate()
        .map(|(direction, (position, size))| (
            position,
            size,
            direction as u16
        ))
        .collect()
}

fn overlaps(chunks: &[Chunk], chunk_size: usize) -> Rules {
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
                                    .and_modify(|labels| labels.insert(other_index, 1))
                                    .or_insert({
                                        let mut set = MSu16xNU::empty();
                                        set.insert(other_index, 1);
                                        set
                                    });
                            }
                        })
                });
            acc
        })
}

// Create a raw graph for pruning
fn create_raw_graph(all_labels: &MSu16xNU, chunk_size: usize, (height, width): (usize, usize)) -> Graph {
    // pixel based graph dimensions
    let v_dim_x = (width * chunk_size) - (chunk_size - 1);
    let v_dim_y = (height * chunk_size) - (chunk_size - 1);

    let vertices: Vec<MSu16xNU> = vec![*all_labels; v_dim_x * v_dim_y];

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
            range_cart_prod
                .clone()
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

    Graph::new(vertices, edges, *all_labels)
}

fn propagate_overlaps(mut graph: Graph, rules: &Rules, label: usize) -> Graph {
    let central_vertex = (graph.vertices.len() - 1) / 2;
    graph.vertices.index_mut(central_vertex).choose(label);
    collapse::collapse(rules, graph, None, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::hash_map;
    use image::ImageBuffer;
    use std::ops::{Index, Not};
    use std::iter::FromIterator;
    use std::convert::TryInto;

    #[test]
    fn test_alias_pixels() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);
        assert_eq!(pixel_aliases.len(), 4);
    }

    #[test]
    fn test_chunk_image() {
        let img = image::open("resources/test/chunk_image_test.png").unwrap().to_rgb8();
        let mut pixel_aliases: PixelKeys = BiMap::new();
        pixel_aliases.insert(0, Rgb::from([255, 255, 255]));
        pixel_aliases.insert(1, Rgb::from([0, 0, 0]));

        let chunk_map = chunk_image(img, 2, &pixel_aliases, true);

        let mut expected_map: LinkedHashMap<Chunk, usize> = LinkedHashMap::new();
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![1, 0, 0, 0]), 1);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![0, 1, 0, 0]), 1);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![0, 0, 1, 0]), 1);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![0, 0, 0, 1]), 1);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![0, 1, 1, 1]), 2);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![1, 0, 1, 1]), 2);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![1, 1, 0, 1]), 2);
        expected_map.insert(DMatrix::from_row_slice(2, 2, &vec![1, 1, 1, 0]), 2);

        assert_eq!(chunk_map.len(), 8);
        expected_map
            .iter()
            .for_each(|(chunk, frequency)| {
                assert_eq!(chunk_map.get(chunk).unwrap(), frequency);
            });
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

        let mut overlaps_n2: Rules = HashMap::new();
        overlaps_n2.insert((5, 0), [0, 1, 0, 0].iter().collect());
        overlaps_n2.insert((0, 1), [1, 0, 0, 0].iter().collect());
        overlaps_n2.insert((6, 1), [1, 0, 0, 0].iter().collect());
        overlaps_n2.insert((1, 0), [0, 1, 0, 0].iter().collect());
        overlaps_n2.insert((2, 1), [1, 0, 0, 0].iter().collect());
        overlaps_n2.insert((7, 0), [0, 1, 0, 0].iter().collect());
        overlaps_n2.insert((2, 2), [0, 1, 0, 0].iter().collect());
        overlaps_n2.insert((5, 1), [0, 0, 1, 0].iter().collect());

        let result_n2 = overlaps(&chunks_n2, 2);
        assert_eq!(result_n2, overlaps_n2);

        let chunks_n3 = vec![
            DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
            DMatrix::from_row_slice(3, 3, &vec![9, 10, 11, 12, 13, 14, 15, 16, 0])
        ];

        let mut overlaps_n3: Rules = HashMap::new();
        overlaps_n3.insert((0, 0), [0, 1].iter().collect());
        overlaps_n3.insert((23, 1), [1, 0].iter().collect());

        let result_n3 = overlaps(&chunks_n3, 3);

        assert_eq!(result_n3, overlaps_n3);

        let chunks_n4 = vec![
            DMatrix::from_row_slice(4, 4, &vec![0, 0, 2, 3,
                                                0, 1, 4, 5,
                                                6, 7, 0, 0,
                                                8, 9, 0, 1])
        ];

        // test overlapping with self only
        let mut overlaps_n4: Rules = HashMap::new();
        overlaps_n4.insert((8, 0), [1, 0].iter().collect());
        overlaps_n4.insert((39, 0), [1, 0].iter().collect());

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

        let mut all_labels = MSu16xNU::empty();
        for i in 0..chunks_n3.len() {
            all_labels.insert(i, 1)
        }

        let raw_graph = create_raw_graph(&all_labels, 3, (2, 2));

        assert_eq!(raw_graph.edges.get(&0).unwrap(), edges_n3.get(&0).unwrap());
        assert_eq!(raw_graph.edges.get(&1).unwrap(), edges_n3.get(&1).unwrap());
        assert_eq!(raw_graph.edges.get(&2).unwrap(), edges_n3.get(&2).unwrap());
        assert_eq!(raw_graph.edges.get(&3).unwrap(), edges_n3.get(&3).unwrap());
        assert_eq!(raw_graph.edges.get(&4).unwrap(), edges_n3.get(&4).unwrap());
    }

    #[test]
    fn test_multiset() {
        let mut x: MSu16xNU = MSu16xNU::empty();
        x.insert(0, 10);
        x.insert(2, 3);

        x.into_iter().enumerate().for_each(|(i, v)| {
            if v != 0 {
                println!("{} {}", i, v);
            }
        })
    }

    #[test]
    fn match_vec() {
        let x = vec![0, 1, 2, 3];
        let y: Vec<usize> = Vec::new();

        let x_prime = x.is_empty().not().then(|| x);
        match x_prime {
            Some(list) => list.iter().for_each(|z| println!("{}", z)),
            None => println!("Empty")
        }
        // let y_prime =
    }

    #[test]
    fn pixels_test() {
        let mut x: Rgb<u8> = Rgb::from([125, 125, 125]);
        // Rgb::
        // Rgb::from_channels(x.channels().iter().map(|q| q+10).collect());
        // let r = (x.channels()[0]);
        // Rgb::fr
        // let y = Rbg::from_channels(r + 10, g + 10, b + 10);
        // let v = x
        //     .channels()
        //     .into_iter()
        //     .map(|q| q+10)
        //     .collect::<Vec<u8>>().as_slice();
        let c: [u8; 3] = x
            .channels()
            .into_iter()
            .map(|q| q+10)
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();

        let y = Rgb::from(c);
        // let y: Rgb<u8> = Rgb::from(x
            // .channels()
            // .into_iter()
            // .map(|q| q+10)
            // .collect::<Vec<u8>>().try_into::<[u8; 3]>().unwrap());
        // x.channels()[0] = [150, 150, 150];
        // let mut p = [0, 0, 0];
        // x.channels().iter().for_each(|q| q+10).collect::<[u32; 3]>()
        // let y = Rgb::from();
        println!("{:?}", y.channels());
        // println!("{:?}", v);

    }

    #[test]
    fn test_vec_slice() {
        let mut x: Rgb<u8> = Rgb::from([125, 125, 125]);
        let y = Rgb::from_slice(&(*x
                              .channels()
                              .into_iter()
                              .map(|q| q+10)
                              .collect::<Vec<u8>>()
            .into_boxed_slice()));
    }

    #[test]
    fn blend_vec() {
        let x = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    }

    #[test]
    fn map_channels() {
        let mut x: Rgb<u8> = Rgb::from([125, 125, 125]);
        x = x.map(|channel| channel + 10);

        println!("{:?}", x);
    }

    #[test]
    fn mut_channels() {
        let mut x: Rgb<u8> = Rgb::from([125, 125, 125]);
        let mut c = x.channels_mut();
        let y = [10, 10, 20];
        // c = c.iter().zip(y.iter()).map(|(p_1, p_2)| p_1 + p_2).collect();
        for i in 0..c.len() {
            c[i] += y[i]
        }
        // c[0] += 10;
        println!("{:?}", x);
    }

    #[test]
    fn re_mut_channels() {
        let mut x: Rgb<u8> = Rgb::from([125, 125, 125]);
        let y = [10, 10, 20];
        let mut c = x.channels_mut();
        c
            .into_iter()
            .zip(y.into_iter())
            .for_each(|(p_1, p_2)| {
                *p_1 += p_2
            });

        println!("{:?}", x);
    }

    #[test]
    fn map_dmatrix() {
        let x = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

        let y: DMatrix<Rgb<usize>> = x.iter().map(|v| Rgb::from([*v, *v, *v])).collect();

        y.iter().for_each(|r| println!("{:?}", r));
        // println!("{}", y);
    }
}

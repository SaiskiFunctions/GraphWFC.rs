use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, Rgb, RgbImage};
use itertools::{Itertools};
use nalgebra::{DMatrix};
use std::collections::HashSet;
// TODO: Change to absolute paths?
use super::tri_wave::u_tri_wave;
use super::limit_iter::limit_iter;
use super::sub_matrix::SubMatrix;
use super::super::graph::graph::{Rules, Edges, Graph};
use super::super::multiset::Multiset;
use num_traits::One;

// Matrix and image data is in COLUMN MAJOR so:
// [1, 2, 3, 4] is equivalent to:
/*
1 3
2 4
*/

// static RGB_CHANNELS: u8 = 3;

// TODO: Implement parse function that will act like OLM Main
pub fn parse() {
}

fn sub_images(image: RgbImage, chunk_size: u32) -> impl Iterator<Item = RgbImage> {
    let height_iter = 0..image.dimensions().0 - (chunk_size - 1);
    let width_iter = 0..image.dimensions().1 - (chunk_size - 1);
    height_iter
        .cartesian_product(width_iter)
        .map(move |(y, x)| imageops::crop_imm(&image, x, y, chunk_size, chunk_size).to_image())
}

pub trait Rotation {
    fn rotate_90(&self) -> DMatrix<u32>;
}

impl Rotation for DMatrix<u32> {
    fn rotate_90(&self) -> DMatrix<u32> {
        assert_eq!(self.nrows(), self.ncols());
        let side = self.nrows();
        let mut target_matrix = DMatrix::<u32>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| target_matrix[(j, (side - 1) - i)] = self[(i, j)]);
        });

        target_matrix
    }
}

fn alias_sub_image(image: RgbImage, pixel_aliases: &BiMap<u32, Rgb<u8>>) -> Vec<u32> {
    image
        .pixels()
        .map(|p| *pixel_aliases.get_by_right(&p).unwrap())
        .collect()
}

fn alias_pixels(image: &RgbImage) -> BiMap<u32, Rgb<u8>> {
    image
        .pixels()
        .fold(HashSet::<Rgb<u8>>::new(), |mut acc, pixel| {
            acc.insert(*pixel);
            acc
        })
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as u32, p))
        .collect()
}

fn chunk_image(
    image: RgbImage,
    chunk_size: u32,
    pixel_aliases: &BiMap<u32, Rgb<u8>>,
) -> HashSet<DMatrix<u32>> {
    sub_images(image, chunk_size)
        .map(|sub_image| alias_sub_image(sub_image, pixel_aliases))
        .fold(HashSet::new(), |mut acc, pixels| {
            let chunk = DMatrix::from_row_slice(chunk_size as usize, chunk_size as usize, &pixels);

            acc.insert(chunk.clone());

            let chunk_r90 = chunk.rotate_90();
            acc.insert(chunk_r90.clone());

            let chunk_r180 = chunk_r90.rotate_90();
            acc.insert(chunk_r180.clone());

            let chunk_r270 = chunk_r180.rotate_90();
            acc.insert(chunk_r270);
            acc
        })
}

pub trait PureReverse<T>
where T: Clone
{
    fn pure_reverse(self) -> Vec<T>;
}

impl<T> PureReverse<T> for Vec<T>
where T: Clone
{
    fn pure_reverse(self) -> Vec<T> {
        let mut vec_rev = self.clone();
        vec_rev.reverse();
        vec_rev
    }
}

// ================================================================================================
// ================================================================================================
// ================================================================================================

/*
    1. Convert chunk_set to a map with indexes
        Each chunk needs to be assigned a label (this is the same as a VertexLabel)

    2. Compute basic overlaps of chunks in the 8 cardinal and inter-cardinal directions:
        a. Generate sub chunk positions
        b. create an accessor to get Matrix chunks
    3. Compute overlaps in a list of overlap structs (N-1 number of overlapping steps) -> but only 1 overlapping map
    3. Use the overlap struct to create a connections struct
 */

// for the case of N=3 we still just connect via cardinals

// This can just be a graph
// type ChunkIndex = i32;
// struct Chunk {
//     pub pixels: DMatrix<u32>,
//     pub connections:
// }

// a function that returns a list of 4D tuples that contain sub chunk positions
// we don't care about the case where the overlap is the entire chunk
// There are period^2 - 1 tuples per chunk

type Position = (u32, u32);
type Size = (u32, u32);
type Direction = u16;
fn sub_chunk_positions(chunk_size: u32) -> Vec<(Position, Size, Direction)> {
    let period = ((chunk_size * 2) - 1) as usize;
    limit_iter(chunk_size)
        .zip(u_tri_wave(chunk_size))
        .take(period)
        .cartesian_product(
            limit_iter(chunk_size)
                .zip(u_tri_wave(chunk_size))
                .take(period))
        .map(|((y_position, y_size), (x_position, x_size))| (
            (x_position, y_position),
            (x_size + 1,  y_size + 1)
        ))
        .filter(|(_,(width, height))| width != &chunk_size || height != &chunk_size)
        .enumerate()
        .map(|(direction, (position, size))| (
            position,
            size,
            direction as u16
        ))
        .collect()
}

fn set_to_map<T>(set: HashSet<T>) -> HashMap<u32, T> {
    set
        .into_iter()
        .enumerate()
        .fold(HashMap::<u32, T>::new(), |mut acc, (k, v)| {
            acc.insert(k as u32, v);
            acc
        })
}

// TODO: Intermediate step that converts the result of chunk_image to a vec so that chunks are labelled
// TODO: Generate implicit linked chunks automatically
// TODO: Change hashset to map
// TODO: Change to rules format
fn overlaps<S: Multiset>(chunks: Vec<DMatrix<u32>>, chunk_size: u32) -> Rules<S> {
    chunks
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (index, chunk)| {
            sub_chunk_positions(chunk_size)
                .into_iter() // equivalent to de-referencing
                .for_each(|((x, y), (width, height), direction)| {
                    let sub_chunk = chunk.sub_matrix((x, y), (width, height));
                    chunks
                        .iter()
                        .enumerate()
                        .for_each(|(other_index, other_chunk)| {
                            // reverse to find mirrored sub chunk
                            let ((o_x, o_y), (o_width, o_height), _) =
                                sub_chunk_positions(chunk_size)
                                .pure_reverse()[direction as usize];
                            let other_sub_chunk = other_chunk
                                .sub_matrix((o_x, o_y), (o_width, o_height));
                            if sub_chunk == other_sub_chunk {
                                acc
                                    .entry((direction, index as usize))
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

// TODO: Generic version of Multiset for Rules used?
// type TempRules = HashMap<(u16, usize), HashSet<>>
//                                 Vertex Label
//                                      |
//                                      V             V
// fn overlaps_to_rules<S: Multiset>(overlaps: HashMap<u32, HashSet<(u32, u32)>>) -> Rules<S> {
//     overlaps
//         .iter()
//         .fold(HashMap::new(), | mut acc, (label, (overlap, direction)) | {
//             acc
//                 .entry((*direction, label as usize))
//                 .and_modify(|labels| *labels.index_mut(*overlap) = One::one())
//                 .or_insert({
//                     let mut set = S::empty(overlaps.len());
//                     *set.index_mut(*overlap) = One::one();
//                     set
//                 });
//             acc
//         })
// }

// Create the raw set of rules
// don't need to do this step as we already basically have created the rules
fn raw_rules() {

}

// Create a raw graph for pruning
fn create_raw_graph<S: Multiset>(chunks: Vec<DMatrix<u32>>, chunk_size: u32, (height, width): (u32, u32)) -> Graph<S> {
    // pixel based graph dimensions
    let v_dim_x = (width * chunk_size) - (chunk_size - 1);
    let v_dim_y = (height * chunk_size) - (chunk_size - 1);

    let mut all_labels = S::empty(chunks.len());
    for i in 0..chunks.len() {
        *all_labels.index_mut(i) = One::one()
    }

    let mut vertices: Vec<S> = Vec::new();
    for _ in 0..(v_dim_x * v_dim_y) {
        vertices.push(all_labels.clone())
    }

    let edges: Edges = vertices
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (index, _)| {
            let (x, y) = index_to_coords(index as u32, (v_dim_x, v_dim_y));
            // create negative indexed range to offset vertex centered directional field by N
            let range = (0 - (chunk_size as i32 - 1))..(chunk_size as i32);
            range
                .clone()
                .cartesian_product(range)
                // remove 0 offset for correct directional mapping
                .filter(|(y_offset, x_offset)| *y_offset != 0 || *x_offset != 0)
                // calculate real cartesian space offest coordinates
                .map(|(y_offset, x_offset)| (y as i32 + y_offset, x as i32 + x_offset))
                .enumerate()
                // remove coordinates outside of graph
                .filter(|(direction, (y_offset, x_offset))| is_inside((*x_offset, *y_offset), (v_dim_x, v_dim_y)))
                .for_each(|(direction, (y_offset, x_offset))| {
                    let other_index = coords_to_index((x_offset as u32, y_offset as u32),
                                                      (v_dim_x, v_dim_y));
                    acc
                        .entry(index as u32)
                        .and_modify(|v| v.push((other_index, direction as u16)))
                        .or_insert(vec![(other_index, direction as u16)]);
                });
            acc
        });

    Graph::new(vertices, edges, all_labels)
}
//
// fn create_raw_edges(length: u32) { // -> Edges {
//     (0..length)
//         .for_each(|index| {
//             let (x, y) = index_to_coords(index as u32, (v_dim_x, v_dim_y));
//             // create negative indexed range to offset vertex centered directional field by N
//             let range = (0 - (chunk_size - 1))..(chunk_size);
//             range
//                 .clone()
//                 .cartesian_product(range)
//                 // remove 0 offset for correct directional mapping
//                 .filter(|v, w| v == 0 && w == 0)
//                 .enumerate()
//                 .for_each(|(direction, (x_offset, y_offset))| {
//                     let other_index = coords_to_index((x + x_offset, y + y_offset),
//                                                       (v_dim_x, v_dim_y));
//                     edges
//                         .entry(index as u32)
//                         .and_modify(|v| v.push((other_index, direction as u16)))
//                         .or_insert(vec![(other_index, direction as u16)]);
//                 })
//         })
// }

fn index_to_coords(index: u32, (w, h): (u32, u32)) -> (u32, u32) {
    (index % w, index / h)
}

fn coords_to_index((x, y): (u32, u32), (w, h): (u32, u32)) -> u32 { x + y * h }

fn is_inside((x, y): (i32, i32), (w, h): (u32, u32)) -> bool {
    if x < 0 || y < 0 || x > (w as i32 -1) || y > (h as i32 -1) { false } else { true }
}

// what structure does this actually return?
// for each chunk it should return the chunk and a list of its connections (and the direction?)
// so find all the intermediate connections -> do we need to know the direction for that?
//
// fn overlaps_2(chunk_set: HashSet<DMatrix<u32>>, chunk_size: u32) { //-> Vec<(DMatrix<u32>, Vec<DMatrix<u32>>)>
//     // give each chunk in the set an Id
//     // let chunk_list = set_to_map::<DMatrix<u32>>(chunk_set);
//     let chunk_map = set_to_map(chunk_set);
//     // overlapping based on sub_chunks
//     // chunk_set
//     //     .iter()
//     //     .for_each(|chunk| {
//     //         let scp = sub_chunk_positions(chunk, chunk_size);
//     //         chunk_set
//     //             .iter()
//     //             .filter(|x| x != chunk)
//     //             .map(|x| x.rotate_90().rotate_90())
//     //             .for_each(|overlap| {
//     //                 chunk.overlaps?(overlap, scp);
//     //                 //
//     //             })
//     //     });
//
//     // return x;
// }

// image overlaps? fn
// fn overlaps(
//     target: &DMatrix<u32>,
//     overlap: &DMatrix<u32>,
//     target_position: (u32, u32),
//     overlap_position: (u32, u32),
//     chunk_size: u32,
// ) -> Bool {
//     // 2D iteration through a chunk
//     (0..chunk_size)
//         .cartesian_product(0..chunk_size)
//         // assume overlap is possible
//         .fold(true, |acc, (y, x)| {
//             // if in a non negative space
//             if not_negative_coord(y - overlap_position.0, x - overlap_position.1) {
//                 // compare the value at the current REAL SPACE position of target
//                 // to the current position in overlap
//                 if target[(y - overlap_position.0, x - overlap_position.1)] != overlap[(y, x)] {
//                     acc = false; // if they don't overlap assumption is falsified
//                 }
//             }
//             acc
//         })
//     // translate overlap matrix into target matrix space
//     // reference overlaps by same index coordiante system
//     // compare the values
//
//     // calculate translation values
//     // translation: (i32, i32)
//
//     // 0, 1 + 0, 0 2-0 = 2 2-1 = 1
//     // 1, 0, 1, 1
//     // overlap.get_pixel(
// }

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;
    use std::iter::FromIterator;
    use nalgebra::{Vector4, Vector2, VectorN, U87, Vector1};
    use crate::utils::hash_map;
    use std::hash::Hash;

    #[test]
    fn test_alias_pixels() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);
        assert_eq!(pixel_aliases.len(), 4);
    }

    #[test]
    fn test_move_data() {
        let side = 4;
        let basic_matrix = DMatrix::from_column_slice(
            side,
            side,
            &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        let mut target_matrix = DMatrix::<u32>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| target_matrix[(j, (side - 1) - i)] = basic_matrix[(i, j)]);
        });
    }

    #[test]
    fn test_chunk_image() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);

        let chunk_set = chunk_image(img, 2, &pixel_aliases);

        let chunk = chunk_set.iter().next().unwrap();

        println!("{:?}", chunk);

        assert_eq!(chunk_set.len(), 4);
        assert!(chunk_set.contains(&chunk.rotate_90()));
        assert!(chunk_set.contains(&chunk.rotate_90().rotate_90()));
        assert!(chunk_set.contains(&chunk.rotate_90().rotate_90().rotate_90()));
    }

    #[test]
    fn what_sort_of_matrix_is_this() {
        let v = vec![0, 1, 2, 3];
        let matrix = DMatrix::from_row_slice(2, 2, &v);
        println!("{}", matrix);
        //                     y, x
        //                    (column position, row position)
        println!("{}", matrix[(1, 0)]);
    }

    #[test]
    fn test_set_to_map() {
        let set = HashSet::from_iter(vec![10, 11, 12, 13]);

        let map = set_to_map(set);
        let values: Vec<_> = map.values().collect();

        // confirm that hashset values were correctly submitted to map›
        assert!(values.iter().any(|&x| x == &10));
        assert!(values.iter().any(|&x| x == &11));
        assert!(values.iter().any(|&x| x == &11));
        assert!(values.iter().any(|&x| x == &11));
    }

    #[test]
    fn test_subchunk_positions() {
        let sub_chunks = vec![
            ((0, 0), (1, 1), 0),
            ((0, 0), (2, 1), 1),
            ((1, 0), (1, 1), 2),
            ((0, 0), (1, 2), 3),
            //         ((0, 0), (2, 2), 4) --> Implicit full overlap removed
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

        let result_n2 = overlaps(chunks_n2, 2);
        assert_eq!(result_n2, overlaps_n2);

        let chunks_n3 = vec![
            DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
            DMatrix::from_row_slice(3, 3, &vec![9, 10, 11, 12, 13, 14, 15, 16, 0])
        ];

        chunks_n3
            .iter()
            .for_each(|c| println!("{}", c));

        let mut overlaps_n3: Rules<Vector2<u32>> = HashMap::new();
        overlaps_n3.insert((0, 0), Multiset::from_row_slice_u(&[0, 1]));
        overlaps_n3.insert((23, 1), Multiset::from_row_slice_u(&[1, 0]));

        let result_n3 = overlaps(chunks_n3, 3);

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

        let results_n4 = overlaps(chunks_n4, 4);

        assert_eq!(results_n4, overlaps_n4);
    }

    #[test]
    fn test_index_to_coords() {
        assert_eq!(index_to_coords(4, (3, 3)), (1, 1));
        assert_eq!(index_to_coords(4, (4, 4)), (0, 1));
        assert_eq!(index_to_coords(11, (3, 5)), (2, 4));
    }

    #[test]
    fn test_coords_to_index() {
        assert_eq!(coords_to_index((2, 1), (3, 3)), 5);
        assert_eq!(coords_to_index((0, 1), (4, 4)), 4);
    }

    #[test]
    fn test_is_inside() {
        assert!(!is_inside((-1, 0), (3, 3)));
        assert!(!is_inside((0, 4), (4, 4)));
    }

    #[test]
    fn test_create_raw_graph() {

        let chunks_n3 = vec![
            DMatrix::from_row_slice(1, 1, &vec![0])
        ];

        let mut edges_n3: Edges = hash_map(&[
            (0, vec![(1, 12), (2, 13), (4, 16), (5, 17), (6, 18), (8, 21), (9, 22), (10, 23)]),
            (1, vec![(0, 11), (2, 12), (3, 13), (4, 15), (5, 16), (6, 17), (7, 18), (8, 20), (9, 21), (10, 22), (11, 23)]),
            (2, vec![(0, 10), (1, 11), (3, 12), (4, 14), (5, 15), (6, 16), (7, 17), (8, 19), (9, 20), (10, 21), (11, 22)]),
            (3, vec![(1, 10), (2, 11), (5, 14), (6, 15), (7, 16), (9, 19), (10, 20), (11, 21)]),
            (4, vec![(0, 7), (1, 8), (2, 9), (5, 12), (6, 13), (8, 16), (9, 17), (10, 18), (12, 21), (13, 22), (14, 23)]),
            // (5, vec![(4, 11), ()])
        ]);

        let raw_graph = create_raw_graph::<Vector1<u32>>(chunks_n3, 3, (2, 2));

        assert_eq!(raw_graph.edges.get(&0).unwrap(), edges_n3.get(&0).unwrap());
        assert_eq!(raw_graph.edges.get(&1).unwrap(), edges_n3.get(&1).unwrap());
        assert_eq!(raw_graph.edges.get(&2).unwrap(), edges_n3.get(&2).unwrap());
        assert_eq!(raw_graph.edges.get(&3).unwrap(), edges_n3.get(&3).unwrap());
        assert_eq!(raw_graph.edges.get(&4).unwrap(), edges_n3.get(&4).unwrap());
    }
}

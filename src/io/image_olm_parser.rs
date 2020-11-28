use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use itertools::{Itertools,kmerge};
use nalgebra::geometry::Rotation2;
use nalgebra::{DMatrix, Matrix2};
use std::collections::HashSet;
use num_traits::{Zero, One};
use std::f32::consts::PI;
use super::tri_wave::u_tri_wave;
use super::limit_iter::limit_iter;
// use std::f32::sin;

// Matrix and image data is in COLUMN MAJOR so:
// [1, 2, 3, 4] is equivalent to:
/*
1 3
2 4
*/
// fn gib_my_values(&self) -> impl Iterator<Item = Foo>

static RGB_CHANNELS: u8 = 3;

pub fn parse() {
    let img = image::open("resources/test/City.png").unwrap();
}

fn sub_images(image: RgbImage, chunk_size: u32) -> impl Iterator<Item = RgbImage> {
    let height_iter = (0..image.dimensions().0 - (chunk_size - 1));
    let width_iter = (0..image.dimensions().1 - (chunk_size - 1));
    height_iter
        .cartesian_product(width_iter)
        .map(move |(y, x)| imageops::crop_imm(&image, x, y, chunk_size, chunk_size).to_image())
}

// pub trait Chunk {
//     type Iter: Iterator<Item = RgbImage>;
//     fn sub_images(&self, chunk_size: u32) -> Self::Iter;
// }
//
// impl Chunk for RgbImage {
//     type Iter = impl Iterator<Item = RgbImage>;
//     fn sub_images(&self, chunk_size: u32) -> Self::Iter {
//         let height_iter = (0..self.dimensions().0 - (chunk_size - 1));
//         let width_iter = (0..self.dimensions().1 - (chunk_size - 1));
//         height_iter.cartesian_product(width_iter)
//             .map(|(y, x)| imageops::crop_imm(self, x, y, chunk_size, chunk_size).to_image())
//         // imageops::crop_imm(self, x, y, chunk_size, chunk_size).to_image()
//     }
// }

// pub trait Chunk {
//     fn sub_images(&self, chunk_size: u32) -> self::Iter;
// }

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

// alias_image
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

// ================================================================================================
// ================================================================================================
// ================================================================================================


trait SubMatrix {
    fn crop_left(self, offset: usize) -> DMatrix<u32>;
    fn crop_right(self, offset: usize) -> DMatrix<u32>;
    fn crop_top(self, offset: usize) -> DMatrix<u32>;
    fn crop_bottom(self, offset: usize) -> DMatrix<u32>;
    fn sub_matrix(&self, position: (usize, usize), size: (usize, usize)) -> DMatrix<u32>;
}

impl SubMatrix for DMatrix<u32> {
    fn crop_left(self, offset: usize) -> DMatrix<u32> {
        self.remove_columns(0, offset)
    }

    fn crop_right(self, offset: usize) -> DMatrix<u32> {
        let cols = self.ncols();
        if offset >= cols { return self }
        self.remove_columns(offset, cols - offset)
    }

    fn crop_top(self, offset: usize) -> DMatrix<u32> {
        self.remove_rows(0, offset)
    }

    fn crop_bottom(self, offset: usize) -> DMatrix<u32> {
        let rows = self.nrows();
        if offset >= rows { return self }
        self.remove_rows(offset, rows - offset)
    }

    fn sub_matrix(&self, position: (usize, usize), size: (usize, usize)) -> DMatrix<u32> {
        self
            .clone()
            .crop_left(position.0)
            .crop_right(size.0)
            .crop_top(position.1)
            .crop_bottom(size.1)
    }
}

//       ┏   x                    IF: x < base -1
// f(x)  ┫
//       ┗  -x + (base * 2 -2)    IF: x >= base
fn wave_sequence(base: i32, x: i32) -> i32 {
    let period = (base * 2) - 2;
    let position = x % period;
    if position < (base - 1) { return position }
    -position + period
}

fn limit_sequence(base: i32, x: i32) -> i32 {
    if x < (base -1) { return 0 }
    x - (base - 1)
}

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
type Direction = u32;
fn sub_chunk_positions(chunk_size: u32) -> Vec<(Position, Size, Direction)> {
    let period = ((chunk_size * 2) - 1) as usize;
    u_tri_wave(chunk_size)
        .zip(limit_iter(chunk_size))
        .take(period)
        .cartesian_product(
            u_tri_wave(chunk_size)
                .zip(limit_iter(chunk_size))
                .take(period))
        .enumerate()
        .map(|(direction, ((y_wave, y_limit), (x_wave, x_limit)))| (
            (x_limit, y_limit),
            (x_wave + 1,  y_wave + 1),
            direction as u32
        ))
        .filter(|(_,(w, h),_)| w != &chunk_size || h != &chunk_size) // TODO: Re-zero index directions
        .collect()


    // let period = (chunk_size * 2) - 1;
    // (0..period)
    //     .cartesian_product(0..period)
    //     .enumerate()
    //     .map(|(direction, (y, x))| (
    //         (limit_sequence(chunk_size as i32, x as i32) as u32, limit_sequence(chunk_size as i32, y as i32) as u32),
    //         ((wave_sequence(chunk_size as i32, x as i32) + 1) as u32, (wave_sequence(chunk_size as i32, y as i32) + 1) as u32),
    //         direction as u32
    //     ))
    //     .filter(|(_,(w, h),_)| w != &chunk_size || h != &chunk_size)
    //     .collect()
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
// what structure does this actually return?
// for each chunk it should return the chunk and a list of its connections (and the direction?)
// so find all the intermediate connections -> do we need to know the direction for that?

fn overlaps_2(chunk_set: HashSet<DMatrix<u32>>, chunk_size: u32) { //-> Vec<(DMatrix<u32>, Vec<DMatrix<u32>>)>
    // give each chunk in the set an Id
    // let chunk_list = set_to_map::<DMatrix<u32>>(chunk_set);
    let chunk_map = set_to_map(chunk_set);
    // overlapping based on sub_chunks
    // chunk_set
    //     .iter()
    //     .for_each(|chunk| {
    //         let scp = sub_chunk_positions(chunk, chunk_size);
    //         chunk_set
    //             .iter()
    //             .filter(|x| x != chunk)
    //             .map(|x| x.rotate_90().rotate_90())
    //             .for_each(|overlap| {
    //                 chunk.overlaps?(overlap, scp);
    //                 //
    //             })
    //     });

    // return x;
}

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

// naive the connections

// prunes connections
// PartialOrd --> uses >= and <=

fn positive<T>(x: T) -> bool
    where T: Zero + PartialOrd
{
    return x >= Zero::zero()
}

// fn significant<T>(x: T) -> bool
//     where T: One + PartialOrd
// {
//     return x >= One::one()
// }
//
// fn significant_2<T>(x: T) -> bool
//     where T: Zero + Ord
// {
//     return x > Zero::zero()
// }
//
// fn not_negative_t<T>(x: T) -> bool
//     where T: Ord
// {
//     x > -1
// }
//
// fn not_negative_coord(x: u32, y: u32) -> bool {
//     return not_negative(x) && not_negative(y)
// }

#[cfg(test)]
mod tests {

    use super::*;
    use image::ImageBuffer;
    use std::iter::FromIterator;

    // #[test]
    // fn test_sub_chunk_positions() {
    //     let side = 2;
    //     let basic_matrix = DMatrix::from_column_slice(
    //         side,
    //         side,
    //         &vec![1, 2, 3, 4]
    //     );
    //     let expected = vec![
    //         (0, 0, 1, 1),
    //         (0, 1, 1, 1),
    //         (1, 0, 1, 1),
    //         (1, 1, 1, 1),
    //         (0, 0, 2, 1),
    //         (0, 1, 2, 1),
    //         (0, 0, 1, 2),
    //         (1, 0, 1, 2),
    //         (0, 0, 2, 2)
    //     ];
    //
    //     let result = sub_chunk_positions(&basic_matrix, side as u32);
    //
    //     //assert_eq!()
    // }

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
    fn test_rotation_matrix() {
        // let basic_matrix = Matrix2::from_row_slice(&vec![2, 3, 4, 5]);
        // let rotation_matrix = Matrix2::from_row_slice(&vec![0, -1, 1, 0]);
        // let rotated_matrix = basic_matrix * rotation_matrix;

        // just alias your RGB channels
        let basic_matrix = DMatrix::from_row_slice(2, 2, &vec![2, 3, 4, 5]);
        let rotation_matrix = DMatrix::from_row_slice(2, 2, &vec![0, -1, 1, 0]);
        let rotated_matrix = basic_matrix * rotation_matrix;
        //println!("{:?}", rotated_matrix);
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

    // how do you test that this works?
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
    fn d_matrix_columns() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let matrix = DMatrix::from_row_slice(3, 3, &v);
        println!("{}", matrix);
        let x = matrix.ncols();
        println!("{:?}", matrix.remove_columns(10, 0));
        // println!("{}", matrix.column_part(2, 2));
        // println!("{}", matrix.row_part(2, 3).remove_columns(0, 2));
        // println!("{}", matrix.clone().remove_columns(0, 1).remove_rows(0, 1));
        // println!("{}", matrix.clone().remove_columns(1, 2).remove_rows(2, 1));
        // println!("{}", matrix.clone().remove_columns(0, 1).remove_columns(2, 1).remove_rows(0, 1).remove_rows(2, 1));
        // println!("{}", matrix.remove_columns(0, 0).remove_columns(2, 1).remove_rows(0, 1).remove_rows(1, 1));
        // (0..2).for_each(|| {
        // });
    }

    #[test]
    fn test_sub_matrix() {
        let v = vec![0, 1, 2,
                     3, 4, 5,
                     6, 7, 8];
        //                                   ┏━━━━━ Height
        //                                   ┃
        //                                   ┃  ┏━━ Width
        //                                   V  V
        let matrix = DMatrix::from_row_slice(3, 3, &v);

        let x = vec![4, 5];
        let target_a = DMatrix::from_row_slice(1, 2, &x);

        let x = vec![1, 2, 4, 5];
        let target_b = DMatrix::from_row_slice(2, 2, &x);

        let x = vec![0, 1, 2];
        let target_c = DMatrix::from_row_slice(1, 3, &x);

        let x = vec![4];
        let target_d = DMatrix::from_row_slice(1, 1, &x);

        //                                    ┏━━━━━ Width
        //                                    ┃
        //                                    ┃  ┏━━ Height
        //                                    V  V
        assert_eq!(matrix.sub_matrix((0, 0), (3, 3)), matrix);

        assert_eq!(matrix.sub_matrix((1, 1), (2, 1)), target_a);

        assert_eq!(matrix.sub_matrix((1, 0), (2, 2)), target_b);

        assert_eq!(matrix.sub_matrix((0, 0), (3, 1)), target_c);

        assert_eq!(matrix.sub_matrix((1, 1), (1, 1)), target_d);
    }

    #[test]
    fn test_seq_wave() {
        // let result_base_3: Vec<i32> = (0..8).map(|x| seq_straight(3, x)).collect();
        // assert_eq!(result_base_3, vec![1, 2, 3, 2, 1, 2, 3, 2]);
        //
        // let result_base_4: Vec<i32> = (0..8).map(|x| seq_straight(4, x)).collect();
        // assert_eq!(result_base_4, vec![1, 2, 3, 4, 3, 2, 1, 2]);
        // println!("{}", 3 % 5);
        // println!("{}", seq_straight(3, 5))

        let result_base_3: Vec<i32> = (0..8).map(|x| wave_sequence(3, x)).collect();
        assert_eq!(result_base_3, vec![0, 1, 2, 1, 0, 1, 2, 1]);
    }

    #[test]
    fn test_limit_sequence() {
        let result_base_3: Vec<i32> = (0..5).map(|x| limit_sequence(3, x)).collect();
        assert_eq!(result_base_3, vec![0, 0, 0, 1, 2]);
    }

    #[test]
    fn test_subchunks() {
        sub_chunk_positions(2).iter().for_each(|x| println!("{:?}", x));
        //     vec![
//         ((0, 0), (1, 1), 0),
//         ((0, 0), (2, 1), 1),
//         ((1, 0), (1, 1), 2),
//         ((0, 0), (1, 2), 3),
// //      ((0, 0), (2, 2), X) --> Implicit full overlap removed
//         ((1, 0), (1, 2), 4),
//         ((0, 1), (1, 1), 5),
//         ((0, 1), (2, 1), 6),
//         ((1, 1), (1, 1), 7)
//     ]

        // ((0, 0), (1, 1), 0)
        //     ((0, 0), (2, 1), 1)
        //     ((1, 0), (1, 1), 2)
        //     ((0, 0), (1, 2), 3)
        //     ((1, 0), (1, 2), 5)
        //     ((0, 1), (1, 1), 6)
        //     ((0, 1), (2, 1), 7)
        //     ((1, 1), (1, 1), 8)
    }

    #[test]
    fn test_cartesian() {
        let x_iter = (10..20);
        let y_iter = (30..40);
        // x_iter.cartesian_product(y_iter).for_each(|(y, x)| println!("{} and {}", y, x))
        x_iter.cartesian_product(y_iter).enumerate().for_each(|(index, (x, y))| println!("{} and {} and {}", index, x, y));
    }

    #[test]
    fn double_iterator() {
        let first_iterator = (0..5);
        let second_iterator = (5..10);
        // first_iterator.chain(second_iterator).for_each(|x| println!("{}", x));
        // first_iterator.combine_with(second_iterator); // -> (0, 5), (1, 6), (2, 7)]]
        // for elt in kmerge(vec![vec![0, 2, 4], vec![1, 3, 5], vec![6, 7]]) {
        //     println!("{}", elt)
        // }
        // first_iterator.zip(second_iterator).for_each(|x| println!("{:?}", x));
        let v = first_iterator.zip(second_iterator).collect::<Vec<(usize, usize)>>();
        // let v: Vec<(usize, usize)> = first_iterator.zip(second_iterator).collect();
        println!("{:?}", v);
    }

    #[test]
    fn double_wave() {
        println!("Here is some \n Split")
        // u_tri_wave(3)
        //     .zip(limit_iter(3))
        //     .take(5)
        //     .cartesian_product(u_tri_wave(3)
        //         .zip(limit_iter(3))
        //         .take(5))
        //     .enumerate()
        //     .for_each(|(index, (x, y))| println!("{:?} and {:?} and {}", x, y, index));
    }
}

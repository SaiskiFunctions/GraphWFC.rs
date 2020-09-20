use image::{ImageBuffer, GenericImageView, DynamicImage, RgbImage, Rgb, imageops};
use nalgebra::{DMatrix, Matrix2};
use nalgebra::geometry::Rotation2;
use std::collections::HashSet;
use hashbrown::HashMap;
use bimap::BiMap;
use itertools::Itertools;

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

fn sub_images(image: RgbImage, chunk_size: u32) -> impl Iterator<Item=RgbImage> {
    let height_iter = (0..image.dimensions().0 - (chunk_size - 1));
    let width_iter = (0..image.dimensions().1 - (chunk_size - 1));
    height_iter.cartesian_product(width_iter)
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
            (0..side).for_each(|j| {
                target_matrix[(j, (side - 1) - i)] = self[(i, j)]
            });
        });

        target_matrix
    }
}

fn alias_sub_image(image: RgbImage, pixel_aliases: &BiMap<u32, Rgb<u8>>) -> Vec<u32> {
    image.pixels().map(|p| *pixel_aliases.get_by_right(&p).unwrap()).collect()
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

fn chunk_image(image: RgbImage, chunk_size: u32, pixel_aliases: &BiMap<u32, Rgb<u8>>) -> HashSet<DMatrix<u32>> {
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

// image overlaps? fn
fn overlaps(target: &DMatrix<u32>, overlap: &DMatrix<u32>, target_position: (u32, u32), overlap_position: (u32, u32), chunk_size: u32) -> Bool {
    // translate overlap matrix into target matrix space
    // reference overlaps by same index coordiante system
    // compare the values

    // calculate translation values
    // translation: (i32, i32)

    // 0, 1 + 0, 0 2-0 = 2 2-1 = 1
    // 1, 0, 1, 1
    // overlap.get_pixel(
}

// naive the connections


// prunes connections

#[cfg(test)]
mod tests {

    use super::*;
    use image::ImageBuffer;
    use std::iter::FromIterator;

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
        let basic_matrix = DMatrix::from_column_slice(side, side, &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut target_matrix = DMatrix::<u32>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| {
                target_matrix[(j, (side - 1) - i)] = basic_matrix[(i, j)]
            });
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
}



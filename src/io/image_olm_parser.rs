use image::{GenericImageView, DynamicImage, RgbImage};
use std::collections::HashSet;

pub fn parse() {
    let img = image::open("resources/test/City.png").unwrap();
    // 
    // chunk {
    //     vec![4]
    // }
}

fn chunk_image(image: &DynamicImage) -> HashSet<RgbImage> {
    let tile_size = 2;
    let chunk_size = 1; // pixels per tile sector

    let (height, width) = image.dimensions();

    let mut chunk_set: HashSet<RgbImage> = HashSet::new();

    (0..height - (tile_size - 1)).for_each(|y| {
        (0..width - (tile_size - 1)).for_each(|x| {
            chunk_set.insert(image.crop_imm(x, y, tile_size, tile_size).to_rgb());
        });
    });

    chunk_set
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_image_loading() {
        let img = image::open("resources/test/City.png");
        assert!(img.is_ok())
    }

    #[test]
    fn test_image_bytes() {
        let img = image::open("resources/test/City.png").unwrap();
        let img_rgb = img.into_rgb();
        // img_rgb.pixels().for_each(|pixel| println!("{:?}", pixel));
    }

    #[test]
    fn test_image_crop() {
        let img = image::open("resources/test/City.png").unwrap();
        println!("{:?}", img.crop_imm(100, 0, 2, 2).to_rgb());
    }

    #[test]
    fn test_chunk_image() {
        let img = image::open("resources/test/City.png").unwrap();
        let chunk_set = chunk_image(&img);
        println!("{:?}", chunk_set);
    }
}
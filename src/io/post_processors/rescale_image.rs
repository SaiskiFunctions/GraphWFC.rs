use image::RgbImage;
use crate::io::post_processors::post_processor::PostProcessor;
use crate::utils::index_to_coords;
use itertools::Itertools;

pub struct RescaleImage {
    scale: usize,
}

impl RescaleImage {
    pub fn new(scale: usize) -> RescaleImage { RescaleImage { scale } }
}

impl PostProcessor<RgbImage> for RescaleImage {
    fn process(&self, input: &RgbImage) -> RgbImage {
        let scale = self.scale;
        let (width, height) = input.dimensions();
        let mut scaled_img: RgbImage = image::ImageBuffer::new(width * scale as u32,
                                                               height * scale as u32);
        input
            .pixels()
            .enumerate()
            .for_each(|(index, pixel)| {
                let (x, y) = index_to_coords(index, width as usize);
                let top_left_output_x = x * scale;
                let top_left_output_y = y * scale;
                (0..scale)
                    .cartesian_product(0..scale)
                    .for_each(|(offset_x, offset_y)| {
                        let output_x = (top_left_output_x + offset_x) as u32;
                        let output_y = (top_left_output_y + offset_y) as u32;
                        scaled_img.put_pixel(output_x, output_y, *pixel)
                    })
            });

        scaled_img
    }
}
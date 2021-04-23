use image::{Rgb, Pixel, Primitive};
use num::ToPrimitive;
use num_traits::zero;
use itertools::izip;

// pub trait PixelOperations {
//     fn add<'a, T: Primitive>(&mut self, x: Rgb<T>);
// }
//
// impl PixelOperations for Rgb<usize> {
//     fn add<T: Primitive + 'static>(&mut self, x: Rgb<T>)
//     {
//         self
//             .channels_mut()
//             .iter_mut()
//             .zip(x.channels())
//             .for_each(|(self_channel, x_channel)| {
//                 *self_channel += ToPrimitive::to_usize(x_channel).unwrap();
//             })
//     }
// }

fn add_pixels<D, T>(x: Rgb<D>, y: Rgb<T>) -> Rgb<D>
where
    D: Primitive + 'static,
    T: Primitive + ToPrimitive + 'static
{
    let mut output_pixel = create_generic_pixel::<D>();

    izip!(
        output_pixel.channels_mut().iter_mut(),
        x.channels().iter(),
        y.channels().iter()
    ).for_each(|(output_channel, x_channel, y_channel)| {
        let y_channel_d = D::from(*y_channel).unwrap();
        *output_channel = *x_channel + y_channel_d
    });

    output_pixel
}

fn create_generic_pixel<D: Primitive + 'static>() -> Rgb<D>
where
    D: Primitive + 'static
{
    let slice: [D; 3] = [zero::<D>(); 3];
    Rgb::from(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_pixels() {
        let small_pixel = Rgb::<u8>::from([100, 20, 5]);
        let mut large_pixel = Rgb::<usize>::from([10, 40, 1000]);
        let combined_pixel = add_pixels(large_pixel, small_pixel);
        assert_eq!(combined_pixel.channels()[0], 110);
        assert_eq!(combined_pixel.channels()[1], 60);
        assert_eq!(combined_pixel.channels()[2], 1005);
    }
}
use std::f32::consts::PI;

struct tri_wave {
    period: f32,
    curr: i32
}

impl Iterator for tri_wave {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let amplitude = self.period / 4.0;
        let position =
            ((2.0 * -amplitude) / PI)
            * f32::asin(f32::sin(
                ((2.0 * PI / self.period) * (self.curr as f32)) + 2.0 / PI
            ))
            + amplitude;
        self.curr += 1;
        Some(position.round() as u32)
    }
}

fn tri_wave(base: u32) -> tri_wave {
    tri_wave { period: ((base * 2) -2) as f32, curr: 0 }
}

fn i_tri_wave(base: u32, start: i32) -> tri_wave {
    tri_wave { period: ((base * 2) -2) as f32, curr: start }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_wave() {
        tri_wave(4).take(20).for_each(|x| println!("{}", x));
    }

    #[test]
    fn test_i_tri_wave() {
        i_tri_wave(4, -10).take(20).for_each(|x| println!("{}", x));
    }
}
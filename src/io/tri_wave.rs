use std::f32::consts::PI;

struct ITriWave {
    period: f32,
    curr: i32
}

impl Iterator for ITriWave {
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

// Creates a integer triangle wave iterator that can return values for ALL values of x
fn i_tri_wave(base: u32, start: i32) -> ITriWave {
    ITriWave { period: ((base * 2) -2) as f32, curr: start }
}

struct UTriWave {
    base: i32,
    period: i32,
    curr: i32
}

impl Iterator for UTriWave {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let position = self.curr % self.period;
        self.curr+=1;
        if position < (self.base - 1) { return Some(position as u32) }
        Some((-position + self.period) as u32)
    }
}

// Creates a integer triangle wave iterator that can calculates values for positive values of x
fn u_tri_wave(base: u32) -> UTriWave {
    let period = ((base * 2) -2) as i32;
    UTriWave { base: (base as i32), period, curr: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i_tri_wave() {
        let wave = i_tri_wave(3, -5).take(10).collect::<Vec<u32>>();
        // not 100% sure this is returning the correct values based on the graph
        assert_eq!(wave, vec![2, 1, 0, 1, 2, 1, 0, 1, 2, 1]);
    }

    #[test]
    fn test_u_tri_wave() {
        let wave = u_tri_wave(3).take(10).collect::<Vec<u32>>();
        assert_eq!(wave, vec![0, 1, 2, 1, 0, 1, 2, 1, 0, 1]);
    }
}
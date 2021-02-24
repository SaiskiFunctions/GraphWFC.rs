#[derive(Clone, Copy)]
pub struct TriWave {
    base: i32,
    period: i32,
    curr: i32,
}

impl TriWave {
    pub fn new(base: u32) -> TriWave {
        let period = ((base * 2) - 2) as i32;
        TriWave { base: (base as i32), period, curr: 0 }
    }
}

//       ┏   x                    IF: x < base -1
// f(x)  ┫
//       ┗  -x + (base * 2 -2)    IF: x >= base
impl Iterator for TriWave {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let position = self.curr % self.period;
        self.curr += 1;
        if position < (self.base - 1) { return Some(position as u32); }
        Some((-position + self.period) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_wave() {
        let wave = TriWave::new(3).take(10).collect::<Vec<u32>>();
        assert_eq!(wave, vec![0, 1, 2, 1, 0, 1, 2, 1, 0, 1]);
    }
}
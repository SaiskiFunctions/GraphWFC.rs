#[derive(Clone, Copy)]
pub struct TriWave {
    base: usize,
    period: usize,
    curr: usize,
}

impl TriWave {
    pub fn new(base: usize) -> TriWave {
        let period = base * 2 - 2;
        TriWave { base, period, curr: 0 }
    }
}

//       ┏  x                   IF: x < base
// f(x)  ┫
//       ┗  (base * 2 - 2) - x  IF: x >= base
impl Iterator for TriWave {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let position = self.curr % self.period;
        self.curr += 1;
        if position < self.base {
            Some(position)
        } else {
            Some(self.period - position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_wave() {
        let wave = TriWave::new(3).take(10).collect::<Vec<_>>();
        assert_eq!(wave, vec![0, 1, 2, 1, 0, 1, 2, 1, 0, 1]);
    }
}
#[derive(Clone)]
pub struct Limit {
    curr: i32,
    base: i32
}

impl Iterator for Limit {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.curr += 1;
        if self.curr < (self.base - 1) { return Some(0) }
        Some((self.curr - (self.base - 1)) as u32)
    }
}

pub fn limit_iter(base: u32) -> Limit {
    Limit { base: (base as i32), curr: -1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_iter() {
        let wave = limit_iter(3).take(10).collect::<Vec<u32>>();
        assert_eq!(wave, vec![0, 0, 0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
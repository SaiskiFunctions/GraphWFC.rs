struct wave_iter {
    base: i32,
    period: i32,
    curr: i32
}

impl Iterator for wave_iter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let position = self.curr % self.period;
        self.curr+=1;
        if position < (self.base - 1) { return Some(position as u32) }
        Some((-position + self.period) as u32)
    }
}

fn wave_iter(base: u32) -> wave_iter {
    let period = ((base * 2) -2) as i32;
    wave_iter { base: (base as i32), period, curr: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        wave_iter(3).take(20).for_each(|x| println!("{}", x));
    }
}
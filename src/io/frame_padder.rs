pub fn pad_frame(radix: usize, index: usize) -> String {
    if index > radix { panic!("Frame index greater than pad amount.") }
    let base = index.to_string();
    let pad = radix.to_string().len() - base.len();
    "0".repeat(pad) + &base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_frame() {
        let pad = pad_frame(100, 5);
        assert_eq!(pad, "005");

        let pad = pad_frame(1000, 230);
        assert_eq!(pad, "0230");

        let pad = pad_frame(10000, 99998);
        assert_eq!(pad, "99998");

        // let pad = pad_frame(10000, 999989);
        // assert_eq!(pad, "99998");
    }
}
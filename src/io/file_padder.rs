fn pad_file(radix: usize, index: usize) -> String {
    let base = index.to_string();
    let pad = radix.to_string().len() - base.len();
    "0".repeat(pad) + &base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_file() {
        let pad = pad_file(100, 5);
        assert_eq!(pad, "005");

        let pad = pad_file(1000, 230);
        assert_eq!(pad, "0230");

        let pad = pad_file(10000, 99998);
        assert_eq!(pad, "99998");
    }
}
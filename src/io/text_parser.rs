use std::fs::read_to_string;
use std::io::Error;
use crate::wfc::graph::Graph;

// 1. Load file into string
//

pub fn parse(filename: &str) -> Result<String, Error> {
    read_to_string(filename).map(|string| {
        string
    })
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::io::text_parser::parse;

    fn _env() {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("resources/test");
    }

    #[test]
    fn test_read() {
        println!("hello");
        match parse("resources/test/emoji.txt") {
            Ok(string) => println!("{}", string),
            Err(e) => println!("hi {}", e)
        };
    }
}

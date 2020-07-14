use crate::graph::{VertexIndex, EdgeDirection};

#[derive(Debug)]
pub struct Propagate {
    from: VertexIndex,
    to: VertexIndex,
    direction: EdgeDirection
}

impl Propagate {
    pub fn new(from: VertexIndex, to: VertexIndex, direction: EdgeDirection) -> Propagate {
        Propagate { from, to, direction }
    }
}

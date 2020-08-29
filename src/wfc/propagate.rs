use crate::graph::graph::{VertexIndex, EdgeDirection};


#[derive(Debug, Clone)]
pub struct Propagate {
    pub from: VertexIndex,
    pub to: VertexIndex,
    pub direction: EdgeDirection
}

impl Propagate {
    pub fn new(from: VertexIndex, to: VertexIndex, direction: EdgeDirection) -> Propagate {
        Propagate { from, to, direction }
    }
}

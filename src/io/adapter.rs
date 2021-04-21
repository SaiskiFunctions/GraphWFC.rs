pub trait Adapter {
    fn renderer(&self) -> Renderer;
    fn parser(&self) -> Parser;
}

pub trait Renderer {
    fn render(&self);
    fn progress(&self);
}

pub trait Parser {
    fn parse(&self);
}
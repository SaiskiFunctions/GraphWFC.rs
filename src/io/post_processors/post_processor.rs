pub trait PostProcessor<T> {
    fn process(self, input: &T) -> T;
}
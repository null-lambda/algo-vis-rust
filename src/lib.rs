mod utils;
mod voronoi;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use voronoi::geometry::Point;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    let mut rng = SmallRng::from_entropy();
    let mut builder = voronoi::builder::Builder::new();
    let n = 20;
    let mut points = vec![];
    for _ in 0..n {
        let x: i32 = rng.gen_range(0..100);
        let y: i32 = rng.gen_range(0..100);
        points.push(Point::new(x as f64, y as f64));
    }

    builder.add_points(points);
    let s = format!("{:?}", builder.graph.topo);

    builder.run();
    alert(&s);
}

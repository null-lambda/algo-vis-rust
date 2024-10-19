use crate::utils::console_log;
use crate::voronoi::builder::Builder;
use crate::voronoi::geometry;
use crate::voronoi::geometry::Point;
use std::borrow::BorrowMut;
use std::io::Write;
use std::iter;
use wasm_bindgen::prelude::*;

struct Parabola {
    a: f64,
    p: f64,
    q: f64,
}

impl Parabola {
    fn from_focus_directrix(focus: Point<f64>, directrix: f64) -> Self {
        let a = 1.0 / (2.0 * (focus[1] - directrix));
        let p = focus[0];
        let q = (focus[1] + directrix) / 2.0;
        Self { a, p, q }
    }

    fn eval(&self, x: f64) -> f64 {
        self.a * (x - self.p).powi(2) + self.q
    }

    fn to_svg(&self, x_range: [f64; 2]) -> String {
        let [x0, x1] = x_range;
        let y0 = self.eval(x0);
        let y1 = self.eval(x1);
        if !self.a.is_finite() {
            return "".to_string();
        }

        let p0 = Point::new(x0, y0);
        let p1 = Point::new(x1, y1);
        let dp0 = Point::new(1.0, 2.0 * self.a * (x0 - self.p));
        let dp1 = Point::new(1.0, 2.0 * self.a * (x1 - self.p));
        if let Some(p2) = geometry::line_intersection(p0, dp0, p1, dp1) {
            console_log!("{:?}", &(self.p, self.q, self.a));
            format!(
                r#"<path d="M {} {} Q {} {} {} {}" fill="none" stroke="blue"/>"#,
                x0, y0, p2[0], p2[1], x1, y1
            )
        } else {
            format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="blue"  stroke-dasharray="2 1"/>"#,
                x0, y0, x1, y1
            )
        }
    }
}

#[wasm_bindgen]
pub struct ViewModel {
    bbox: [f64; 4],
    model: Builder,
}

#[wasm_bindgen]
impl ViewModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            bbox: [0.0, 0.0, 100.0, 100.0],
            model: Builder::new(),
        }
    }

    pub fn add_point(&mut self, px: f64, py: f64) {
        self.model.add_points(iter::once(Point::new(px, py)));
    }

    fn get_points(&self) -> &[Point<f64>] {
        &self.model.graph.face_center
    }

    fn line(&self, svg: &mut Vec<u8>, p0: Point<f64>, p1: Point<f64>, style: &str) {
        write!(
            svg,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" {style} />"#,
            p0[0], p0[1], p1[0], p1[1],
        )
        .unwrap();
    }

    fn render_beachlines(&self, svg: &mut Vec<u8>) {
        let mut sites_inorder: Vec<Point<f64>> = vec![];
        let mut current: Option<&Box<_>> = self.model.beachline.root.as_ref();
        let mut stack: Vec<&Box<_>> = vec![];
        while current.is_some() || !stack.is_empty() {
            while let Some(node) = current {
                stack.push(node);
                current = node.children[0].as_ref();
            }
            current = stack.pop();

            let node = current.unwrap();
            let point = self.model.graph.face_center[node.value.site_right];
            sites_inorder.push(point);

            current = current.unwrap().children[1].as_ref();
        }

        for p in &sites_inorder {
            write!(
                svg,
                r#"<circle cx="{}" cy="{}" r="1.1" fill="red" />"#,
                p[0], p[1],
            )
            .unwrap();
        }

        let mut directrix = self.model.directrix.into();
        directrix += 1e-9;

        for i in 0..sites_inorder.len() {
            let site = &sites_inorder[i];
            let x0 = (i > 0)
                .then(|| &sites_inorder[i - 1])
                .inspect(|prev| console_log!("prev {:?}, site {:?}", prev, site))
                .map(|prev| geometry::breakpoint_x(*prev, *site, directrix))
                .unwrap_or(self.bbox[0]);
            let x1 = (i < sites_inorder.len() - 1)
                .then(|| &sites_inorder[i + 1])
                .inspect(|next| console_log!("site {:?}, next {:?}", site, next))
                .map(|next| geometry::breakpoint_x(*site, *next, directrix))
                .unwrap_or(self.bbox[2]);
            console_log!("x0: {}, x1: {}", x0, x1);

            let parabola = Parabola::from_focus_directrix(*site, self.model.directrix.into());
            write!(svg, "{}", parabola.to_svg([x0, x1])).unwrap();
            let [x0, y0] = [parabola.p, parabola.q];
            let [x1, y1] = [parabola.p, self.model.directrix.into()];
            self.line(svg, Point::new(x0, y0), Point::new(x1, y1), "stroke='red'");
        }
    }

    pub fn render_to_svg(&self) -> String {
        let mut svg: Vec<u8> = vec![];

        console_log!("{:?}", self.model.beachline);

        write!(
            svg,
            r#"<svg width="100%" height="100%" viewBox="0 0 100 100" style="stroke-width: .5" xmlns="http://www.w3.org/2000/svg">"#
        ).unwrap();

        // sweep line
        let directrix: f64 = self.model.directrix.into();
        write!(
            svg,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" />"#,
            self.bbox[0], directrix, self.bbox[2], directrix,
        )
        .unwrap();

        for p in self.get_points() {
            write!(
                svg,
                r#"<circle cx="{}" cy="{}" r="1" fill="black" />"#,
                p[0], p[1],
            )
            .unwrap();
        }

        self.render_beachlines(&mut svg);

        write!(svg, "</svg>").unwrap();

        String::from_utf8(svg).unwrap()
    }

    pub fn run(&mut self) {
        self.model.run();
    }

    pub fn init(&mut self) {
        self.model.init();
    }

    pub fn step(&mut self) {
        self.model.step();
    }

    pub fn clear(&mut self) {
        self.model = Builder::new();
    }
}

use crate::utils::console_log;
use crate::voronoi::builder::{self, Builder, Event};
use crate::voronoi::cmp::Trivial;
use crate::voronoi::geometry;
use crate::voronoi::geometry::Point;

use std::cmp::Reverse;
use std::collections::HashMap;
use std::io::Write;
use std::iter;
use svg::{Circle, Close, Line, Open};
use wasm_bindgen::prelude::*;

pub mod svg {
    use std::fmt;

    use crate::voronoi::geometry::{self, Point};

    type Style<'a> = &'a str;

    pub struct Open<'a>(pub &'a str);

    impl<'a> fmt::Display for Open<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "<{}>", self.0)
        }
    }

    pub struct Close<'a>(pub &'a str);

    impl<'a> fmt::Display for Close<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "</{}>", self.0)
        }
    }

    #[derive(Clone, Copy)]
    pub struct Line<'a>(pub [f64; 4], pub Style<'a>);

    impl<'a> fmt::Display for Line<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" {} />"#,
                self.0[0], self.0[1], self.0[2], self.0[3], self.1
            )
        }
    }

    #[derive(Clone, Copy)]
    pub struct Circle<'a>(pub [f64; 3], pub Style<'a>);

    impl<'a> fmt::Display for Circle<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                r#"<circle cx="{}" cy="{}" r="{}" {}/>"#,
                self.0[0], self.0[1], self.0[2], self.1
            )
        }
    }

    #[derive(Clone, Copy)]
    pub struct Parabola {
        pub a: f64,
        pub p: f64,
        pub q: f64,
    }

    impl Parabola {
        pub fn from_focus_directrix(focus: Point<f64>, directrix: f64) -> Self {
            let a = 1.0 / (2.0 * (focus[1] - directrix));
            let p = focus[0];
            let q = (focus[1] + directrix) / 2.0;
            Self { a, p, q }
        }

        pub fn eval(&self, x: f64) -> f64 {
            let y = self.a * (x - self.p).powi(2) + self.q;
            if !y.is_finite() {
                return self.q;
            }
            y
        }

        pub fn ranged(self, x_range: [f64; 2], style: Style) -> RangedParabola {
            RangedParabola {
                parabola: self,
                x_range,
                style,
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct RangedParabola<'a> {
        pub parabola: Parabola,
        pub x_range: [f64; 2],
        pub style: Style<'a>,
    }

    impl<'a> fmt::Display for RangedParabola<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let this = &self.parabola;
            let x_range = self.x_range;
            let [x0, x1] = x_range;
            let y0 = this.eval(x0);
            let y1 = this.eval(x1);
            if !this.a.is_finite() {
                return Ok(());
            }

            let p0 = Point::new(x0, y0);
            let p1 = Point::new(x1, y1);
            let dp0 = Point::new(1.0, 2.0 * this.a * (x0 - this.p));
            let dp1 = Point::new(1.0, 2.0 * this.a * (x1 - this.p));
            if let Some(p2) = geometry::line_intersection(p0, dp0, p1, dp1) {
                // console_log!("{:?}", &(self.p, self.q, self.a));
                write!(
                    f,
                    r#"<path d="M {} {} Q {} {} {} {}" fill="none" {}/>"#,
                    x0, y0, p2[0], p2[1], x1, y1, self.style
                )
            } else {
                write!(
                    f,
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke-dasharray="2 1" {}/>"#,
                    x0, y0, x1, y1, self.style
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SiteMarker {
    Preprocess,
    BeachLine,
    Complete,
}
// impl to_str
impl SiteMarker {
    fn to_class(&self) -> &'static str {
        match self {
            Self::Preprocess => "preprocess",
            Self::BeachLine => "beachline",
            Self::Complete => "complete",
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
        if cfg!(debug_assertions) {
            crate::utils::set_panic_hook();
        }

        Self {
            bbox: [-100.0, -100.0, 300.0, 300.0],
            model: Builder::new(),
        }
    }

    pub fn add_point(&mut self, px: f64, py: f64) {
        self.model.add_points(iter::once(Point::new(px, py)));
    }

    fn get_points(&self) -> &[Point<f64>] {
        &self.model.graph.face_center
    }

    fn add_css(&self, svg: &mut Vec<u8>) {
        let css = r#"<style>
 svg {
     stroke-width: .5;
 }
svg .site {
    fill: black;
}
sg .beachline {
    fill: blue;
}
svg .site>.beachline{
    fill:blue;
}
svg .site>.preprocess{
    fill:gray;
}
svg .site>.complete{
    fill:black;
}
svg .parabola {
    stroke: blue;
}
svg .breakpoint {
    stroke: blue;
}
svg .circumcircle {
    fill: none;
    stroke: lightgreen;
}
svg .event-circle {
    fill: lightgreen;
}
svg .sweepline {
    stroke: black;
}
</style>"#;

        write!(svg, "{}", css).unwrap();
    }

    fn gen_site_marker(&self) -> HashMap<usize, SiteMarker> {
        let mut site_marker = HashMap::new();
        for i in 0..self.get_points().len() {
            site_marker.insert(i, SiteMarker::Complete); // todo
        }
        // for
        site_marker
    }

    fn render_sites(&self, svg: &mut Vec<u8>, site_markers: &HashMap<usize, SiteMarker>) {
        write!(svg, "{}", Open("g class='site'")).unwrap();
        for (&i, &marker) in site_markers {
            let p = self.model.graph.face_center[i];
            let s = format!("class='{}'", marker.to_class());
            write!(svg, "{}", Circle([p[0], p[1], 1.0], s.as_str())).unwrap();
        }
        write!(svg, "{}", Close("g")).unwrap();
    }

    fn render_beachlines(&self, svg: &mut Vec<u8>, site_markers: &mut HashMap<usize, SiteMarker>) {
        unsafe {
            let mut sites_inorder: Vec<Point<f64>> = vec![];
            builder::splay::Node::inorder(self.model.beachline, |node| {
                let site = node.as_ref().value.site;
                let point = self.model.graph.face_center[site];
                site_markers.insert(site, SiteMarker::BeachLine);
                sites_inorder.push(point);
            });

            write!(svg, "{}", Open("g class='beachline'")).unwrap();

            let mut directrix: f64 = self.model.directrix.into();
            directrix += 1e-9;

            write!(svg, "{}", Open("g class='parabola'")).unwrap();
            let breakpoints: Vec<f64> = iter::once(self.bbox[0])
                .chain(
                    sites_inorder
                        .windows(2)
                        .map(|t| geometry::breakpoint_x(t[0], t[1], directrix)),
                )
                .chain(iter::once(self.bbox[0] + self.bbox[2]))
                .collect();

            console_log!("{:?}", sites_inorder.iter().map(|p| p).collect::<Vec<_>>());
            console_log!("{:?}", breakpoints);

            for i in 0..sites_inorder.len() {
                let site = &sites_inorder[i];
                let x0 = breakpoints[i];
                let x1 = breakpoints[i + 1];
                let parabola = svg::Parabola::from_focus_directrix(*site, directrix);
                write!(svg, "{}", parabola.ranged([x0, x1], "")).unwrap();
            }
            write!(svg, "{}", Close("g")).unwrap();

            write!(svg, "{}", Open("g class='breakpoint'")).unwrap();
            for i in 1..sites_inorder.len() {
                let site = &sites_inorder[i];
                let x0 = breakpoints[i];
                let parabola = svg::Parabola::from_focus_directrix(*site, directrix);
                let y0 = parabola.eval(x0);
                let r = 2.0;
                write!(svg, "{}", Line([x0, y0 + r, x0, y0 - r], "")).unwrap();
                write!(svg, "{}", Line([x0 + r, y0, x0 - r, y0], "")).unwrap();
            }
            write!(svg, "{}", Close("g")).unwrap();
            write!(svg, "{}", Close("g")).unwrap();
        }
    }

    fn render_circ_events(&self, svg: &mut Vec<u8>, site_markers: &mut HashMap<usize, SiteMarker>) {
        let mut queue = self.model.events.clone();
        write!(svg, "{}", Open("g class='circumcircle'")).unwrap();
        while let Some((Reverse((qy, qx)), Trivial(event))) = queue.pop() {
            match event {
                Event::Circle(circle) => {
                    if !builder::check_circle_event(&circle) {
                        continue;
                    }

                    let prev = unsafe { circle.node.as_ref().side[0].unwrap() };
                    let next = unsafe { circle.node.as_ref().side[1].unwrap() };
                    let p0 =
                        self.model.graph.face_center[unsafe { circle.node.as_ref().value.site }];
                    let p1 = self.model.graph.face_center[unsafe { prev.as_ref().value.site }];
                    let p2 = self.model.graph.face_center[unsafe { next.as_ref().value.site }];
                    let Some(center) = geometry::circumcenter(p0, p1, p2) else {
                        continue;
                    };

                    let qx: f64 = qx.into();
                    let qy: f64 = qy.into();

                    let radius = (p0 - center).norm_sq().sqrt();
                    write!(svg, "{}", Circle([center[0], center[1], radius], "")).unwrap();
                    write!(svg, "{}", Circle([qx, qy, 1.0], "class='event-circle'")).unwrap();
                }
                Event::Site(site) => {
                    site_markers.insert(site.idx, SiteMarker::Preprocess);
                }
            }
        }
        write!(svg, "{}", Close("g")).unwrap();
    }

    fn render_sweepline(&self, svg: &mut Vec<u8>) {
        let directrix: f64 = self.model.directrix.into();
        write!(
            svg,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="sweepline"/>"#,
            self.bbox[0], directrix, self.bbox[2], directrix,
        )
        .unwrap();
    }

    pub fn render_to_svg(&self) -> String {
        let mut svg: Vec<u8> = vec![];

        console_log!("{:?}", unsafe { self.model.beachline.as_ref() });

        let bbox = self.bbox;
        write!(
            svg,
            r#"<svg width="100%" height="100%" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">"#
            , bbox[0], bbox[1], bbox[2], bbox[3]
                    
        ).unwrap();

        let mut site_markers = self.gen_site_marker();
        self.add_css(&mut svg);
        self.render_circ_events(&mut svg, &mut site_markers);
        self.render_beachlines(&mut svg, &mut site_markers);
        self.render_sweepline(&mut svg);
        self.render_sites(&mut svg, &site_markers);

        write!(svg, "</svg>").unwrap();

        String::from_utf8(svg).unwrap()
    }

    pub fn run(&mut self) {
        self.model.run();
    }

    pub fn init(&mut self) {
        self.model.init();
    }

    pub fn step(&mut self) -> bool {
        self.model.step()
    }

    pub fn clear(&mut self) {
        self.model = Builder::new();
    }
}

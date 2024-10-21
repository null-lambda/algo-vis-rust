use crate::voronoi::builder::{self, Builder, Event};
use crate::voronoi::cmp::Trivial;
use crate::voronoi::geometry::Point;
use crate::voronoi::graph::HalfEdge;
use crate::voronoi::{self, geometry};

use std::cmp::Reverse;
use std::collections::HashMap;
use std::io::Write;
use std::iter;
use svg::{Circle, Close, Line, Open};
use wasm_bindgen::prelude::*;

pub mod svg {
    use std::fmt;

    use crate::voronoi::geometry::{self, Point};

    use super::BBox;

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

        pub fn ranged(self, bbox: BBox, x_range: [f64; 2], style: Style) -> RangedParabola {
            RangedParabola {
                bbox,
                parabola: self,
                x_range,
                style,
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct RangedParabola<'a> {
        pub bbox: BBox,
        pub parabola: Parabola,
        pub x_range: [f64; 2],
        pub style: Style<'a>,
    }

    impl<'a> fmt::Display for RangedParabola<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let this = &self.parabola;
            let x_range = self.x_range;
            let [mut x0, mut x1] = x_range;
            x0 = x0.max(self.bbox.x);
            x1 = x1.min(self.bbox.x + self.bbox.w);
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
                write!(
                    f,
                    r#"<path d="M {} {} Q {} {} {} {}" fill="none" {}/>"#,
                    x0, y0, p2[0], p2[1], x1, y1, self.style
                )?;
            } else {
                // let Parabola { p, q, .. } = self.parabola;
                // write!(
                //     f,
                //     r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke-dasharray="2 1" {}/>"#,
                //     p,
                //     q,
                //     p,
                //     q - self.bbox.h * 2.0,
                //     self.style
                // )
            }
            Ok(())
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
#[derive(Clone, Copy, Debug)]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[wasm_bindgen]
impl BBox {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }
}

#[wasm_bindgen]
pub struct ViewModel {
    bbox: BBox,
    model: Builder,
}

#[wasm_bindgen]
impl ViewModel {
    #[wasm_bindgen(constructor)]
    pub fn new(&bbox: &BBox) -> Self {
        if cfg!(debug_assertions) {
            crate::utils::set_panic_hook();
        }

        Self {
            bbox,
            model: Builder::new(),
        }
    }

    pub fn add_point(&mut self, px: f64, py: f64) {
        self.model.add_points(iter::once(Point::new(px, py)));
    }

    pub fn get_num_points(&self) -> usize {
        self.model.graph.face_center.len()
    }

    fn get_points_slice(&self) -> &[Point<f64>] {
        &self.model.graph.face_center
    }

    pub fn get_num_edges(&self) -> usize {
        self.model.graph.topo.half_edges.len()
    }

    pub fn get_points(&self) -> Vec<f64> {
        self.get_points_slice()
            .iter()
            .flat_map(|p| vec![p[0], p[1]])
            .collect()
    }

    fn gen_site_marker(&self) -> HashMap<usize, SiteMarker> {
        let mut site_marker = HashMap::new();
        for i in 0..self.get_points_slice().len() {
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
            let breakpoints: Vec<f64> = iter::once(self.bbox.x)
                .chain(
                    sites_inorder
                        .windows(2)
                        .map(|t| geometry::breakpoint_x(t[0], t[1], directrix)),
                )
                .chain(iter::once(self.bbox.x + self.bbox.w))
                .collect();

            for i in 0..sites_inorder.len() {
                let site = &sites_inorder[i];
                let x0 = breakpoints[i];
                let x1 = breakpoints[i + 1];

                let parabola = svg::Parabola::from_focus_directrix(*site, directrix);
                write!(svg, "{}", parabola.ranged(self.bbox, [x0, x1], "")).unwrap();
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

    fn render_half_edges(&self, svg: &mut Vec<u8>) {
        let voronoi::graph::Graph {
            vert_coord,
            face_center,
            topo: voronoi::graph::Topology { half_edges, .. },
        } = &self.model.graph;

        for &h1 in half_edges {
            let h2 = half_edges[h1.twin];
            let get_p = |h: HalfEdge| {
                if h.vert != voronoi::graph::INF {
                    vert_coord[h.vert]
                } else {
                    let f1 = face_center[h.face_left];
                    let f2 = face_center[half_edges[h.twin].face_left];
                    let mid = (f1 + f2) * 0.5;
                    let mut dir = (f2 - f1).rot();
                    dir = dir * (dir.norm_sq().sqrt() + 1e-9).recip();
                    mid + dir * 5e2
                }
            };
            let p1 = get_p(h1);
            let p2 = get_p(h2);

            writeln!(svg, "{}", Circle([p1[0], p1[1], 1.0], "class='vertex'")).unwrap();
            writeln!(svg, "{}", Circle([p2[0], p2[1], 1.0], "class='vertex'")).unwrap();
            writeln!(
                svg,
                "{}",
                Line([p1[0], p1[1], p2[0], p2[1]], "class='edge-voronoi'")
            )
            .unwrap();
        }

        write!(svg, "{}", Open("g class='edges'")).unwrap();
        for (i_h1, &h1) in half_edges.iter().enumerate() {
            let i_h2 = h1.twin;
            if i_h2 < i_h1 {
                continue;
            }
            let h2 = half_edges[i_h2];

            let p1 = face_center[h1.face_left];
            let p2 = face_center[h2.face_left];

            writeln!(
                svg,
                "{}",
                Line([p1[0], p1[1], p2[0], p2[1]], "class='edge-delaunay'")
            )
            .unwrap();
        }

        write!(svg, "{}", Close("g")).unwrap();
    }

    fn render_sweepline(&self, svg: &mut Vec<u8>) {
        let directrix: f64 = self.model.directrix.into();
        write!(
            svg,
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="sweepline"/>"#,
            self.bbox.x,
            directrix,
            self.bbox.x + self.bbox.w,
            directrix,
        )
        .unwrap();
    }

    pub fn render_to_svg(&self) -> Vec<String> {
        let mut header: Vec<u8> = vec![];
        let mut rest: Vec<u8> = vec![];

        let bbox = self.bbox;
        write!(
            header,
            r#"<svg class="voronoi" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            bbox.x, bbox.y, bbox.w, bbox.h
        )
        .unwrap();

        if !self.get_points_slice().is_empty() {
            let mut site_markers = self.gen_site_marker();
            self.render_half_edges(&mut rest);
            self.render_circ_events(&mut rest, &mut site_markers);
            self.render_beachlines(&mut rest, &mut site_markers);
            self.render_sweepline(&mut rest);
            self.render_sites(&mut rest, &site_markers);
        }

        write!(rest, "</svg>").unwrap();

        let header = String::from_utf8(header).unwrap();
        let rest = String::from_utf8(rest).unwrap();
        vec![header, rest]
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

    pub fn reset(&mut self) {
        let points = self.get_points_slice().to_vec();
        self.model = Builder::new();
        self.model.add_points(points);
    }

    pub fn clear(&mut self) {
        self.model = Builder::new();
    }
}

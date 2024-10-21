use crate::voronoi::builder::{self, Builder, Event};
use crate::voronoi::cmp::Trivial;
use crate::voronoi::geometry::Point;
use crate::voronoi::graph::HalfEdge;
use crate::voronoi::{self, geometry};

use std::cmp::Reverse;
use std::collections::HashMap;
use std::io::Write;
use std::iter;
use wasm_bindgen::prelude::*;

pub mod svg {
    use crate::voronoi::geometry::{self, Point};

    type Style<'a> = &'a str;

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

pub struct SvgRenderer {
    buf: Vec<u8>,
    scale: [f64; 2],
    trans: [f64; 2],
}

impl SvgRenderer {
    pub fn new(bbox: BBox) -> Self {
        debug_assert!(bbox.w > 0.0 && bbox.h > 0.0);
        let scale = [300.0 / bbox.w, 300.0 / bbox.h];
        let trans = [-bbox.x, -bbox.y];
        Self {
            buf: vec![],
            scale,
            trans,
        }
    }

    pub fn emit(&self) -> String {
        unsafe { String::from_utf8_unchecked(self.buf.clone()) }
    }

    fn transform_x(&self, x: f64) -> f64 {
        (x + self.trans[0]) * self.scale[0]
    }

    fn transform_y(&self, y: f64) -> f64 {
        (y + self.trans[1]) * self.scale[1]
    }

    fn transform(&self, x: f64, y: f64) -> (f64, f64) {
        (self.transform_x(x), self.transform_y(y))
    }

    pub fn line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, style: &str) {
        let (x0, y0) = self.transform(x0, y0);
        let (x1, y1) = self.transform(x1, y1);

        write!(
            self.buf,
            "<line x1='{}' y1='{}' x2='{}' y2='{}' {} />",
            x0, y0, x1, y1, style
        )
        .unwrap();
    }

    pub fn cross(&mut self, x: f64, y: f64, r: f64, style: &str) {
        let r = r / self.scale[0];
        self.line(x - r, y - r, x + r, y + r, style);
        self.line(x - r, y + r, x + r, y - r, style);
    }

    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, style: &str) {
        let (cx, cy) = self.transform(cx, cy);
        write!(
            self.buf,
            "<circle cx='{}' cy='{}' r='{}' {} />",
            cx,
            cy,
            r * self.scale[0],
            style
        )
        .unwrap();
    }

    pub fn point(&mut self, x: f64, y: f64, r: f64, style: &str) {
        let (x, y) = self.transform(x, y);
        write!(
            self.buf,
            "<circle cx='{}' cy='{}' r='{}' {} />",
            x, y, r, style
        )
        .unwrap();
    }

    pub fn parabola(&mut self, directrix: f64, focus: Point<f64>, x_range: [f64; 2], style: &str) {
        let focus = Point::new(self.transform_x(focus[0]), self.transform_y(focus[1]));
        let directrix = self.transform_y(directrix);
        let x_range = [self.transform_x(x_range[0]), self.transform_x(x_range[1])];
        let parabola = svg::Parabola::from_focus_directrix(focus, directrix);
        let [mut x0, mut x1] = x_range;
        x0 = x0.max(0.0);
        x1 = x1.min(300.0);
        let y0 = parabola.eval(x0);
        let y1 = parabola.eval(x1);
        if !parabola.a.is_finite() {
            return;
        }

        let p0 = Point::new(x0, y0);
        let p1 = Point::new(x1, y1);
        let dp0 = Point::new(1.0, 2.0 * parabola.a * (x0 - parabola.p));
        let dp1 = Point::new(1.0, 2.0 * parabola.a * (x1 - parabola.p));
        if let Some(p2) = geometry::line_intersection(p0, dp0, p1, dp1) {
            write!(
                self.buf,
                r#"<path d="M {} {} Q {} {} {} {}" fill="none" {}/>"#,
                x0, y0, p2[0], p2[1], x1, y1, style
            )
            .unwrap();
        } else {
            //
        }
    }

    pub fn open(&mut self, tag: &str) {
        write!(self.buf, "<{}>", tag).unwrap();
    }

    pub fn close(&mut self, tag: &str) {
        write!(self.buf, "</{}>", tag).unwrap();
    }
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

    pub fn set_bbox(&mut self, bbox: &BBox) {
        self.bbox = *bbox;
    }

    pub fn get_bbox(&self) -> BBox {
        self.bbox
    }

    pub fn fit_bbox(&mut self) -> BBox {
        if self.get_points_slice().is_empty() {
            return BBox::new(-100.0, -100.0, 300.0, 300.0);
        }

        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in self.get_points_slice() {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        crate::utils::console_log!("{:?} {:?} {:?} {:?}", x0, y0, x1, y1);
        let mut w = (x1 - x0).min(1e300).max(1e-300);
        let mut h = (y1 - y0).min(1e300).max(1e-300);

        if self.get_points_slice().len() == 1 {
            w = 300.0;
            h = 300.0;
        }
        BBox::new(x0 - w * 1., y0 - w * 1., w * 3.0, h * 3.0)
    }

    fn gen_site_marker(&self) -> HashMap<usize, SiteMarker> {
        let mut site_marker = HashMap::new();
        for i in 0..self.get_points_slice().len() {
            site_marker.insert(i, SiteMarker::Complete); // todo
        }
        // for
        site_marker
    }

    fn render_sites(&self, svg: &mut SvgRenderer, site_markers: &HashMap<usize, SiteMarker>) {
        svg.open("g class='site'");
        for (&i, &marker) in site_markers {
            let p = self.model.graph.face_center[i];
            let s = format!("class='{}'", marker.to_class());
            svg.point(p[0], p[1], 1.0, s.as_str());
        }
        svg.close("g");
    }

    fn render_beachlines(
        &self,
        svg: &mut SvgRenderer,
        site_markers: &mut HashMap<usize, SiteMarker>,
    ) {
        unsafe {
            let mut sites_inorder: Vec<Point<f64>> = vec![];
            builder::splay::Node::inorder(self.model.beachline, |node| {
                let site = node.as_ref().value.site;
                let point = self.model.graph.face_center[site];
                site_markers.insert(site, SiteMarker::BeachLine);
                sites_inorder.push(point);
            });

            svg.open("g class='beachline'");

            let mut directrix: f64 = self.model.directrix.into();
            directrix += 1e-9;

            svg.open("g class='parabola'");
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

                svg.parabola(directrix, *site, [x0, x1], "");
            }
            svg.close("g");

            svg.open("g class='breakpoint'");
            for i in 1..sites_inorder.len() {
                let site = &sites_inorder[i];
                let x0 = breakpoints[i];
                let parabola = svg::Parabola::from_focus_directrix(*site, directrix);
                let y0 = parabola.eval(x0);
                svg.cross(x0, y0, 2.0, "");
            }
            svg.close("g");
            svg.close("g");
        }
    }

    fn render_circ_events(
        &self,
        svg: &mut SvgRenderer,
        site_markers: &mut HashMap<usize, SiteMarker>,
    ) {
        let mut queue = self.model.events.clone();
        svg.open("g class='circumcircle'");
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
                    svg.circle(center[0], center[1], radius, "");
                    svg.point(qx, qy, 1.0, "class='event-circle'");
                }
                Event::Site(site) => {
                    site_markers.insert(site.idx, SiteMarker::Preprocess);
                }
            }
        }
        svg.close("g");
    }

    fn render_half_edges(&self, svg: &mut SvgRenderer) {
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
                    mid + dir * (self.bbox.w + self.bbox.h)
                }
            };
            let p1 = get_p(h1);
            let p2 = get_p(h2);

            svg.point(p1[0], p1[1], 1.0, "class='vertex'");
            svg.point(p2[0], p2[1], 1.0, "class='vertex'");
            svg.line(p1[0], p1[1], p2[0], p2[1], "class='edge-voronoi'")
        }

        svg.open("g class='edges'");
        for (i_h1, &h1) in half_edges.iter().enumerate() {
            let i_h2 = h1.twin;
            if i_h2 < i_h1 {
                continue;
            }
            let h2 = half_edges[i_h2];

            let p1 = face_center[h1.face_left];
            let p2 = face_center[h2.face_left];

            svg.line(p1[0], p1[1], p2[0], p2[1], "class='edge-delaunay'");
        }

        svg.close("g");
    }

    fn render_sweepline(&self, svg: &mut SvgRenderer) {
        let directrix: f64 = self.model.directrix.into();
        svg.line(
            self.bbox.x,
            directrix,
            self.bbox.x + self.bbox.w,
            directrix,
            "class='sweepline'",
        );
    }

    pub fn render_to_svg(&self) -> Vec<String> {
        let mut header = SvgRenderer::new(self.bbox);
        let mut rest = SvgRenderer::new(self.bbox);

        header.open(
            format!(
                r#"svg class="voronoi" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg""#,
                0.0, 0.0, 300.0, 300.0
            )
            .as_str(),
        );

        if !self.get_points_slice().is_empty() {
            let mut site_markers = self.gen_site_marker();
            self.render_half_edges(&mut rest);
            self.render_circ_events(&mut rest, &mut site_markers);
            self.render_beachlines(&mut rest, &mut site_markers);
            self.render_sweepline(&mut rest);
            self.render_sites(&mut rest, &site_markers);
        }

        rest.close("svg");

        vec![header.emit(), rest.emit()]
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

import { throttle, sleep, rand_range } from './utils.js';
import * as wasm from "algo-vis-rust";
import * as styles from './index.scss';
import css_voronoi from './voronoi.scss.raw';
const styles_voronoi = `<style>${css_voronoi}</style>`;

let bbox = new wasm.BBox(-100, -100, 300, 300);

let vm = new wasm.ViewModel(bbox);
for (let i = 0; i < 15; i++) {
    const x = rand_range(0, 100000) / 1000;
    const y = rand_range(0, 100000) / 1000;
    vm.add_point(x, y);
}
vm.init();

let paused = false;
let loop = false;
let animation_steps_per_sec = 10;

const elem_play_pause = document.querySelector('#play-pause');
const elem_step = document.querySelector('#step');
const elem_loop = document.querySelector('#loop');
const elem_reset = document.querySelector('#reset');
const elem_clear = document.querySelector('#clear');
const elem_speed = document.querySelector('#speed');
const elem_speed_label = document.querySelector('#speed-label');
const elem_add_points = document.querySelector('#add-random-points');
const elem_num_of_random_points = document.querySelector('#num-of-random-points');
const elem_points_raw = document.querySelector('#points-raw');

const set_paused = value => {
    paused = value;
    elem_play_pause.setAttribute('paused', paused);
};

elem_play_pause.addEventListener('click', () => {
    set_paused(!paused);
});

elem_step.addEventListener('click', () => {
    set_paused(true);
    vm.step();
    render();
});

elem_loop.addEventListener('click', () => {
    loop = elem_loop.checked;
});

elem_reset.addEventListener('click', () => {
    let temp = paused;
    set_paused(true);
    vm.reset();
    vm.init();
    set_paused(temp);
    override_raw_points();
    render();
});

elem_clear.addEventListener('click', () => {
    let temp = paused;
    set_paused(true);
    vm = new wasm.ViewModel(bbox);
    vm.init();
    set_paused(temp);
    override_raw_points();
    render();
});

elem_add_points.addEventListener('click', () => {
    let temp = paused;
    set_paused(temp);
    vm.reset();
    let n_points = parseInt(elem_num_of_random_points.value);
    for (let i = 0; i < n_points; i++) {
        const x = rand_range(0, 1e9) / 1e7;
        const y = rand_range(0, 1e9) / 1e7;
        vm.add_point(x, y);
    }
    vm.init();
    set_paused(temp);
    override_raw_points();
    render();
});

elem_speed.addEventListener('input', _event => {
    const t = parseInt(elem_speed.value) / 100.0;
    const [log_min, log_max] = [-.3, 4];
    animation_steps_per_sec = 10 ** (log_min + t * (log_max - log_min));
    elem_speed_label.innerText = `(expected) ${animation_steps_per_sec.toExponential(1)} steps/s`;
});
elem_speed.dispatchEvent(new PointerEvent('input'));

const override_raw_points = () => {
    const n = vm.get_num_points();
    const points = vm.get_points();

    const buf = [`${n}`];
    for (let i = 0; i < n; i++) {
        const x = points[i * 2];
        const y = points[i * 2 + 1];
        buf.push(`${x} ${y}`);
    }
    elem_points_raw.value = buf.join('\n');
};
override_raw_points();

elem_points_raw.addEventListener('input', throttle(event => {
    const tokens = event.target.value.split(/\s+/);
    const points = [];
    try {
        const n = parseInt(tokens[0]);
        for (let i = 0; i < n; i++) {
            const x = parseFloat(tokens[i * 2 + 1]);
            const y = parseFloat(tokens[i * 2 + 2]);
            if (isNaN(x) || isNaN(y)) {
                throw new Error('NaN');
            }
            points.push([x, y]);
        }
    } catch (e) {
        console.log(e);
        elem_points_raw.setAttribute("parse-error", true);
        return;
    }
    elem_points_raw.setAttribute("parse-error", false);

    let temp = paused;
    set_paused(true);
    vm = new wasm.ViewModel(bbox);
    for (const [x, y] of points) {
        vm.add_point(x, y);
    }
    vm.init();
    set_paused(temp);
    render();
}, 100));


const keymap = {};
for (const elem of document.querySelectorAll('[data-hotkey]')) {
    const key = elem.dataset.hotkey.toUpperCase();
    if (elem.tagName === 'BUTTON'
        || elem.tagName === 'INPUT' && elem.type === 'checkbox') {
        keymap[key] = () => {
            elem.click();
            elem.focus();
        };
    }
}

// hotkeys
document.addEventListener("keydown", event => {
    const key = event.key.toUpperCase();
    if (event.ctrlKey || event.altKey) {
        return;
    }
    if (key in keymap) {
        keymap[key]();
    }
    switch (key) {
        case '+':
        case '=':
            elem_speed.value = Math.min(100, parseInt(elem_speed.value) + 3);
            elem_speed.dispatchEvent(new PointerEvent('input'));
            break;
        case '-':
        case '_':
            elem_speed.value = Math.max(0, parseInt(elem_speed.value) - 3);
            elem_speed.dispatchEvent(new PointerEvent('input'));
            break;
    }
});

const client_to_viewbox = (event) => {
    const u = (event.clientX - event.currentTarget.offsetLeft) / event.currentTarget.offsetWidth;
    const v = (event.clientY - event.currentTarget.offsetTop) / event.currentTarget.offsetHeight;
    const x = bbox.x + bbox.w * u;
    const y = bbox.y + bbox.h * v;
    return [x, y];
};

const elem_canvas = document.querySelector('#canvas');
elem_canvas.addEventListener('click', event => {
    let temp = paused;
    set_paused(true);
    vm.reset();

    const [x, y] = client_to_viewbox(event);
    vm.add_point(x, y);
    vm.init();
    override_raw_points();
    set_paused(temp);
    render();
});


const translate = ([dx, dy]) => bbox => {
    bbox.x += dx;
    bbox.y += dy;
}
const scale_at_origin = (factor) => bbox => {
    bbox.x *= factor;
    bbox.y *= factor;
    bbox.w *= factor;
    bbox.h *= factor;
};
const onZoom = (factor, [mouse_x, mouse_y]) => {
    translate([-mouse_x, -mouse_y])(bbox);
    scale_at_origin(factor)(bbox);
    translate([mouse_x, mouse_y])(bbox);
};
elem_canvas.addEventListener('wheel', event => {
    const factor = 1 + event.deltaY / 1000;
    const [x, y] = client_to_viewbox(event);
    onZoom(factor, [x, y]);
    vm.set_bbox(bbox);
    render();
});

document.addEventListener('resize', () => {
    const w = Math.min(elem_canvas.width, elem_canvas.height);
    elem_canvas.width = w;
    elem_canvas.height = w;
    render();
});


let should_render = false;
const render = () => {
    should_render = true;
};

let t_last_render = Date.now();
const dt_sleep_min = 1000 / 120;
const dt_sleep_max = 1000 * 5;
let dt_sleep = dt_sleep_min;
const render_loop = async () => {
    while (true) {

        const t_start = Date.now();

        if (!should_render) {
            await new Promise(resolve => {
                requestAnimationFrame(resolve);
            });
            continue;
        }

        const [svg_header, svg_rest] = vm.render_to_svg();
        const svg = [svg_header, styles_voronoi, svg_rest].join('\n');
        const domURL = window.URL || window.webkitURL || window;
        console.log(svg);
        const url = domURL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
        const img = new Image();
        await new Promise(resolve => {
            img.addEventListener('load', () => resolve());
            img.src = url;
        });

        const w = Math.min(elem_canvas.clientWidth, elem_canvas.clientHeight);
        elem_canvas.width = w;
        elem_canvas.height = w;

        const ctx = elem_canvas.getContext('2d');
        ctx.canvas.width = elem_canvas.clientWidth;
        ctx.canvas.height = elem_canvas.clientHeight;

        ctx.drawImage(img, 0, 0);
        domURL.revokeObjectURL(url);

        should_render = false;
        console.log('rendered');

        await new Promise(resolve => {
            requestAnimationFrame(resolve);
        });

        const t_end = Date.now();
        const dt_period = t_end - t_last_render;
        const dt_render = t_end - t_start;

        // compare and sleep adaptively to keep render rate less than 1/2 (not to block the main thread)
        const render_percentage = dt_render / dt_period;
        const target_render_percentage = 0.5;
        const adaptivity = 1.25;
        if (render_percentage > target_render_percentage) {
            dt_sleep = Math.min(dt_sleep_max, dt_sleep * adaptivity);
        } else {
            dt_sleep = Math.max(dt_sleep_min, dt_sleep / adaptivity);
        }
        t_last_render = Date.now();
        console.log(`dt_sleep: ${dt_sleep}, dt_render: ${dt_render}, dt_period: ${dt_period}`);

        await sleep(dt_sleep);
    }
};
render_loop();

const step = async () => {
    console.log('step');
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(vm.step());
        }, 1000 / animation_steps_per_sec);
    });
};

const builder_loop = async () => {
    if (!paused) {
        const result = await step();
        if (result) {
            render();
        } else if (loop) {
            elem_reset.dispatchEvent(new PointerEvent('click'));
        }
    }
    requestAnimationFrame(builder_loop);
};
builder_loop();



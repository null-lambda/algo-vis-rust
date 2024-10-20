import * as wasm from "algo-vis-rust";

let seed = 42;
const rand_next = () => {
    seed = ((seed * 166455) + 10139043) % 0xffffffff;
    return seed;
};
const rand_range = (a, b) => {
    return a + rand_next() % (b - a + 1);
};

let vm = new wasm.ViewModel();
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
    render();
});

elem_clear.addEventListener('click', () => {
    let temp = paused;
    set_paused(true);
    vm = new wasm.ViewModel();
    vm.init();
    set_paused(temp);
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
    render();
});


elem_speed.addEventListener('input', _event => {
    const t = parseInt(elem_speed.value) / 100.0;
    const [log_min, log_max] = [-.3, 4];
    animation_steps_per_sec = 10 ** (log_min + t * (log_max - log_min));
    elem_speed_label.innerText = `(expected) ${animation_steps_per_sec.toExponential(1)} steps/s`;
});
elem_speed.dispatchEvent(new PointerEvent('input'));

const throttle = (f, limit) => {
    let inThrottle;
    return (...args) => {
        if (!inThrottle) {
            f(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

// hotkeys
document.addEventListener("keydown", event => {
    switch (event.key.toUpperCase()) {
        case 'P':
            elem_play_pause.click();
            break;
        case 'S':
            elem_step.click();
            break;
        case 'R':
            elem_reset.click();
            break;
        case 'C':
            elem_clear.click();
            break;
        case 'L':
            elem_loop.click();
            break;
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

const elem_canvas = document.querySelector("#canvas");
let svg_screen_ctm = null;

elem_canvas.addEventListener('click', event => {
    console.log(123);
    const elem_svg = elem_canvas.querySelector('svg');
    console.log(elem_svg);
    if (elem_svg) {
        svg_screen_ctm = elem_svg.getScreenCTM();
    }
    if (!svg_screen_ctm) {
        return;
    }

    let temp = paused;
    set_paused(true);
    vm.reset();

    const svg_point = elem_svg.createSVGPoint();
    svg_point.x = event.clientX;
    svg_point.y = event.clientY;

    const svg_coords = svg_point.matrixTransform(svg_screen_ctm.inverse());
    const x = svg_coords.x;
    const y = svg_coords.y;
    console.log(x, y);

    vm.add_point(x, y);
    vm.init();
    set_paused(temp);
    render();
});

const render = throttle(() => {
    requestAnimationFrame(() => {
        elem_canvas.innerHTML = vm.render_to_svg();
    });
}, 1000 / 30);


const step = async () => {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(vm.step());
        }, 1000 / animation_steps_per_sec);
    });
};

const animate = async () => {
    if (!paused) {
        const result = await step();
        if (result) {
            render();
        } else if (loop) {
            elem_reset.dispatchEvent(new PointerEvent('click'));
        }
    }
    requestAnimationFrame(animate);
};
animate();


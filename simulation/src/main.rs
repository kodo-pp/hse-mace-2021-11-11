use nalgebra::{DMatrix, DVector, Matrix3, Matrix3xX, MatrixXx3, Vector3};
use ordered_float::NotNan;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SimulationParams {
    pub p: f64,
    pub q: f64,
    pub r: f64,
    pub c: Matrix3xX<f64>,
}

impl SimulationParams {
    pub fn a(&self) -> Matrix3<f64> {
        Matrix3::from_column_slice(&[-self.p, 0.0, -1.0, 0.0, -self.q, 0.0, 1.0, 0.0, -self.r])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HistoryEntry {
    pub t: f64,
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
    pub u1: f64,
    pub u2: f64,
    pub u3: f64,
}

impl HistoryEntry {
    pub fn x(&self) -> Vector3<f64> {
        Vector3::from_column_slice(&[self.x1, self.x2, self.x3])
    }
}

#[derive(Debug, Clone)]
pub struct History(Vec<HistoryEntry>);

impl History {
    pub fn new(inner: Vec<HistoryEntry>) -> Self {
        Self(inner)
    }

    pub fn sparsify(self, step_size: usize) -> Self {
        Self::new(self.0.into_iter().step_by(step_size).collect())
    }

    pub fn time_to_index(&self, t: f64) -> Option<usize> {
        if self.0.is_empty() {
            return None;
        }

        match self
            .0
            .binary_search_by_key(&NotNan::new(t).unwrap(), |entry| {
                NotNan::new(entry.t).unwrap()
            }) {
            Ok(index) | Err(index) => Some(index.clamp(0, self.0.len() - 1)),
        }
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.0
    }

    pub fn to_csv(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        writeln!(out, "t,x1,x2,x3,u1,u2,u3")?;
        for entry in self.0.iter() {
            writeln!(
                out,
                "{},{},{},{},{},{},{}",
                entry.t, entry.x1, entry.x2, entry.x3, entry.u1, entry.u2, entry.u3,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Simulation {
    params: SimulationParams,
    t: f64,
    x: Vector3<f64>,
    x_history: History,
}

impl Simulation {
    pub fn new(params: SimulationParams, x0: Vector3<f64>) -> Self {
        Self {
            x: x0,
            t: 0.0,
            params,
            x_history: History::new(Vec::new()),
        }
    }

    pub fn x_tuple(&self) -> (f64, f64, f64) {
        (self.x.x, self.x.y, self.x.z)
    }

    pub fn x(&self) -> &Vector3<f64> {
        &self.x
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn step(&mut self, dt: f64, w: &DVector<f64>, u: &Vector3<f64>) {
        let (x1, x2, x3) = self.x_tuple();
        let fx = Vector3::from([x1 * x2, 1.0 - x1.powi(2), 0.0]);
        let dx = self.params.a() * self.x + u + &self.params.c * w + fx;
        let t = self.t();
        let (u1, u2, u3) = (u.x, u.y, u.z);
        self.x_history.0.push(HistoryEntry {
            t,
            x1,
            x2,
            x3,
            u1,
            u2,
            u3,
        });
        self.x += dx * dt;
        self.t += dt;
    }

    pub fn into_x_history(self) -> History {
        self.x_history
    }

    pub fn x_history(&self) -> &History {
        &self.x_history
    }

    pub fn params(&self) -> &SimulationParams {
        &self.params
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Equilibrium {
    Negative,
    Zero,
    Positive,
}

impl Equilibrium {
    pub fn eval(&self, p: f64, q: f64, r: f64) -> Vector3<f64> {
        let sqrt_term = || ((r - q - p * q * r) / r).sqrt();
        let x2_term = || (1.0 + p * r) / r;
        let (x1, x2, x3) = match self {
            Self::Negative => (-sqrt_term(), x2_term(), sqrt_term() / r),
            Self::Zero => (0.0, 1.0 / q, 0.0),
            Self::Positive => (sqrt_term(), x2_term(), -sqrt_term() / r),
        };
        Vector3::from([x1, x2, x3])
    }

    pub fn iter_all() -> impl Iterator<Item = Self> {
        [Self::Negative, Self::Zero, Self::Positive].iter().copied()
    }
}

#[derive(Debug, Clone)]
pub struct ControllerParams {
    pub g: Matrix3xX<f64>,
    pub k1: MatrixXx3<f64>,
    pub k2: MatrixXx3<f64>,
    pub u1: DMatrix<f64>,
    pub u2: DMatrix<f64>,
    pub w1: DMatrix<f64>,
    pub w2: DMatrix<f64>,
    pub equilibrium: Equilibrium,
}

pub trait TauFn {
    fn eval(&mut self, t: f64) -> f64;
    fn mu(&self) -> f64;
    fn tau(&self) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct SineTauFn {
    pub mu: f64,
    pub tau: f64,
    pub phase: f64,
}

impl TauFn for SineTauFn {
    fn eval(&mut self, t: f64) -> f64 {
        self.mu * (t + self.phase).sin() + self.tau - self.mu
    }

    fn mu(&self) -> f64 {
        self.mu
    }

    fn tau(&self) -> f64 {
        self.tau
    }
}

pub trait Controller {
    fn eval(&mut self, sim: &Simulation) -> Vector3<f64>;
}

#[derive(Debug, Clone, Copy)]
pub struct NullController;

impl Controller for NullController {
    fn eval(&mut self, _sim: &Simulation) -> Vector3<f64> {
        Vector3::zeros()
    }
}

#[derive(Debug)]
pub struct ProposedController<'rng, R: Rng, T: TauFn> {
    params: ControllerParams,
    rng: &'rng mut R,
    tau_fn: T,
}

impl<'rng, R: Rng, T: TauFn> ProposedController<'rng, R, T> {
    pub fn new(params: ControllerParams, rng: &'rng mut R, tau_fn: T) -> Self {
        Self {
            params,
            rng,
            tau_fn,
        }
    }

    pub fn gamma1(&mut self) -> DMatrix<f64> {
        self.random_gamma()
    }

    pub fn gamma2(&mut self) -> DMatrix<f64> {
        self.random_gamma()
    }

    pub fn random_gamma(&mut self) -> DMatrix<f64> {
        let (_, rows1) = self.params.u1.shape();
        let (_, rows2) = self.params.u2.shape();
        let (cols1, _) = self.params.w1.shape();
        let (cols2, _) = self.params.w2.shape();
        assert_eq!(rows1, rows2);
        assert_eq!(cols1, cols2);
        assert_eq!(rows1, rows2);
        let size = rows1;
        DMatrix::from_fn(size, size, |r, c| {
            if r == c {
                self.rng.gen_range(0.0..1.0)
            } else {
                0.0
            }
        })
    }
}

impl<R: Rng, T: TauFn> Controller for ProposedController<'_, R, T> {
    fn eval(&mut self, sim: &Simulation) -> Vector3<f64> {
        let sim_params = sim.params();
        let x = sim.x();
        let t = sim.t();
        let hist = sim.x_history();

        let x_star = self
            .params
            .equilibrium
            .eval(sim_params.p, sim_params.q, sim_params.r);

        let gamma1 = self.gamma1();
        let delta_k1 = &self.params.u1 * &gamma1 * &self.params.w1;
        let mult1 = &self.params.g * (&self.params.k1 + &delta_k1);
        let immediate_term = mult1 * (x - x_star);
        let mut result = immediate_term;

        if let Some(back_index) = hist.time_to_index(t - self.tau_fn.eval(t)) {
            let gamma2 = self.gamma2();
            let delta_k2 = &self.params.u2 * &gamma2 * &self.params.w2;
            let mult2 = &self.params.g * (&self.params.k2 + &delta_k2);
            let x_back = hist.entries()[back_index].x();
            let delayed_term = mult2 * (x_back - x_star);
            result += delayed_term;
        }

        if result.norm() >= 10000.0 {
            panic!("overshoot");
        }
        result
    }
}

fn run_simulation(
    params: SimulationParams,
    x0: Vector3<f64>,
    w_fn: &mut impl FnMut(f64) -> DVector<f64>,
    controller: &mut (impl Controller + ?Sized),
    num_steps: u64,
    approx_num_saved_points: u64,
    stop_at: f64,
    enable_controller_at: f64,
    out_path: &(impl AsRef<std::path::Path> + ?Sized),
) -> std::io::Result<()> {
    let dt = stop_at / num_steps as f64;
    let mut sim = Simulation::new(params, x0);
    for _ in 0..num_steps {
        let t = sim.t();
        let w = w_fn(t);
        let u = if t < enable_controller_at {
            Vector3::zeros()
        } else {
            controller.eval(&sim)
        };
        sim.step(dt, &w, &u);
    }
    let hist = sim.into_x_history();
    let out_path = out_path.as_ref();
    let dir = out_path.parent().unwrap();
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let out_file = std::fs::File::create(out_path)?;
    let step_size = if approx_num_saved_points < num_steps {
        (num_steps / approx_num_saved_points) as usize
    } else {
        1
    };
    hist.sparsify(step_size)
        .to_csv(&mut std::io::BufWriter::new(out_file))?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let sim_params_ex1 = SimulationParams {
        p: 2.1,
        q: 0.3,
        r: 1.0,
        c: Matrix3xX::from_iterator(
            3,
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
                .iter()
                .copied(),
        ),
    };
    let ctl_params_ex1 = ControllerParams {
        g: Matrix3xX::from_fn(3, |r, c| if r == c { 0.6 } else { 0.0 }),
        k1: MatrixXx3::from_iterator(
            3,
            [
                -174.1836, -25.4263, -37.8756, //
                -25.4263, -217.2244, -76.8075, //
                -37.8756, -76.8075, -281.6521, //
            ]
            .iter()
            .copied(),
        ),
        k2: MatrixXx3::from_iterator(
            3,
            [
                0.5383, 0.0035, -0.0259, //
                0.0035, 0.5621, 0.0025, //
                -0.0259, 0.0025, 0.3914, //
            ]
            .iter()
            .copied(),
        ),
        u1: DMatrix::from_iterator(3, 1, [0.2, 0.6, 0.4].iter().copied()),
        u2: DMatrix::from_iterator(3, 1, [0.4, 0.3, 0.5].iter().copied()),
        w1: DMatrix::from_iterator(1, 3, [0.4, 0.8, 1.2].iter().copied()),
        w2: DMatrix::from_iterator(1, 3, [0.18, 0.2, 0.7].iter().copied()),
        equilibrium: Equilibrium::Zero,
    };
    let x0_ex1 = Vector3::from([0.1, 0.3, 0.4]);

    let sim_params_ex2 = sim_params_ex1.clone();
    let ctl_params_ex2 = ctl_params_ex1.clone();
    let x0_ex2 = x0_ex1.clone();

    let sim_params_ex3 = SimulationParams {
        p: 2.5,
        q: 0.2,
        r: 1.2,
        c: Matrix3xX::from_iterator(
            3,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                .iter()
                .copied(),
        ),
    };
    let x0_ex3 = x0_ex1.clone();

    let sim_params_ex4 = SimulationParams {
        p: 2.5,
        q: 0.2,
        r: 1.2,
        c: Matrix3xX::from_iterator(1, [1.0, 1.0, 1.0].iter().copied()),
    };
    let ctl_params_ex4a = ControllerParams {
        g: Matrix3xX::from_fn(3, |r, c| if r == c { 0.5 } else { 0.0 }),
        k1: MatrixXx3::from_iterator(
            3,
            [
                -316.9424, -37.0517, -54.9966, //
                -37.0517, -377.5121, -111.1249, //
                -54.9966, -111.1249, -465.5427, //
            ]
            .iter()
            .copied(),
        ),
        k2: MatrixXx3::from_iterator(
            3,
            [
                0.4698, -0.0599, -0.1318, //
                -0.0599, 0.4736, -0.1995, //
                -0.1318, -0.1995, 0.0860, //
            ]
            .iter()
            .copied(),
        ),
        u1: DMatrix::zeros(3, 1),
        u2: DMatrix::zeros(3, 1),
        w1: DMatrix::zeros(1, 3),
        w2: DMatrix::zeros(1, 3),
        //u1: DMatrix::from_iterator(3, 1, [0.2, 0.6, 0.4].iter().copied()),
        //u2: DMatrix::from_iterator(3, 1, [0.4, 0.3, 0.5].iter().copied()),
        //w1: DMatrix::from_iterator(1, 3, [0.4, 0.8, 1.2].iter().copied()),
        //w2: DMatrix::from_iterator(1, 3, [0.18, 0.2, 0.7].iter().copied()),
        equilibrium: Equilibrium::Zero,
    };
    let ctl_params_ex4b = ControllerParams {
        g: Matrix3xX::from_fn(3, |r, c| if r == c { 1.0 } else { 0.0 }),
        k1: MatrixXx3::from_iterator(
            3,
            [
                -151.5625, -109.7926, -118.4274, //
                -111.7692, -153.7464, -115.6126, //
                -108.2663, -110.9650, -157.8783, //
            ]
            .iter()
            .copied(),
        ),
        k2: MatrixXx3::from_iterator(
            3,
            [
                3.4638, 2.6206, 2.6206, //
                2.6206, 3.4638, 2.6206, //
                2.6206, 2.6206, 3.4638, //
            ]
            .iter()
            .copied(),
        ),
        u1: DMatrix::zeros(3, 1),
        u2: DMatrix::zeros(3, 1),
        w1: DMatrix::zeros(1, 3),
        w2: DMatrix::zeros(1, 3),
        //u1: DMatrix::from_iterator(3, 1, [0.2, 0.6, 0.4].iter().copied()),
        //u2: DMatrix::from_iterator(3, 1, [0.4, 0.3, 0.5].iter().copied()),
        //w1: DMatrix::from_iterator(1, 3, [0.4, 0.8, 1.2].iter().copied()),
        //w2: DMatrix::from_iterator(1, 3, [0.18, 0.2, 0.7].iter().copied()),
        equilibrium: Equilibrium::Zero,
    };
    let x0_ex4 = Vector3::from([0.5, 3.0, -0.4]);

    let mut rng = rand::thread_rng();

    // Example 1.
    for with_fluct in [true, false].iter().copied() {
        for (equilibrium, name) in
            Equilibrium::iter_all().zip(["neg", "zero", "pos"].iter().copied())
        {
            println!(
                "[Example 1] Running simulation with{} fluctuations and with equilibrium kind: {:?}",
                if with_fluct { "" } else { "out" },
                equilibrium,
            );
            run_simulation(
                sim_params_ex1.clone(),
                x0_ex1.clone(),
                &mut |t| {
                    if with_fluct {
                        DVector::repeat(3, t.sin() * 0.1)
                    } else {
                        DVector::zeros(3)
                    }
                },
                &mut ProposedController::new(
                    ControllerParams {
                        equilibrium,
                        ..ctl_params_ex1.clone()
                    },
                    &mut rng,
                    SineTauFn {
                        mu: 0.3,
                        tau: 1.3,
                        phase: 0.0,
                    },
                ),
                100000,
                10000,
                200.0,
                100.0,
                &format!(
                    "data-out/ex1/with{}-fluct-eq-{}.csv",
                    if with_fluct { "" } else { "out" },
                    name,
                ),
            )?;
        }
    }

    // Example 2.
    for (with_fluct, with_fault) in [(true, true), (true, false), (false, true), (false, false)]
        .iter()
        .copied()
    {
        for (equilibrium, name) in
            Equilibrium::iter_all().zip(["neg", "zero", "pos"].iter().copied())
        {
            println!(
                "[Example 2] Running simulation with{} fluctuations, with{} fault and with equilibrium kind: {:?}",
                if with_fluct { "" } else { "out" },
                if with_fault { "" } else { "out" },
                equilibrium,
            );
            run_simulation(
                sim_params_ex2.clone(),
                x0_ex2.clone(),
                &mut |t| {
                    if with_fluct {
                        DVector::repeat(3, t.sin() * 0.1)
                    } else {
                        DVector::zeros(3)
                    }
                },
                &mut ProposedController::new(
                    ControllerParams {
                        equilibrium,
                        g: if with_fault {
                            ctl_params_ex2.g.clone()
                        } else {
                            Matrix3xX::from_fn(3, |r, c| if r == c { 1.0 } else { 0.0 })
                        },
                        ..ctl_params_ex2.clone()
                    },
                    &mut rng,
                    SineTauFn {
                        mu: 0.3,
                        tau: 1.3,
                        phase: 0.0,
                    },
                ),
                100000,
                10000,
                0.1,
                0.0,
                &format!(
                    "data-out/ex2/with{}-fluct-with{}-fault-eq-{}.csv",
                    if with_fluct { "" } else { "out" },
                    if with_fault { "" } else { "out" },
                    name,
                ),
            )?;
        }
    }

    #[derive(Debug, Copy, Clone)]
    enum Ex3Fluct {
        None,
        Sine,
        Random,
    }

    // Example 3.
    for with_fluct in [Ex3Fluct::None, Ex3Fluct::Sine, Ex3Fluct::Random]
        .iter()
        .copied()
    {
        println!(
            "[Example 3] Running simulation with{} fluctuations",
            match with_fluct {
                Ex3Fluct::None => "out",
                Ex3Fluct::Sine => " sine",
                Ex3Fluct::Random => " random",
            },
        );
        run_simulation(
            sim_params_ex3.clone(),
            x0_ex3.clone(),
            &mut |t| match with_fluct {
                Ex3Fluct::None => DVector::zeros(3),
                Ex3Fluct::Sine => DVector::repeat(3, 0.1 * t.sin()),
                Ex3Fluct::Random => DVector::from_fn(3, |_, _| rng.gen_range(-0.15..0.15)),
            },
            &mut NullController,
            100000,
            5000,
            500.0,
            0.0,
            &format!(
                "data-out/ex3/with{}-fluct.csv",
                match with_fluct {
                    Ex3Fluct::None => "out",
                    Ex3Fluct::Sine => "-sine",
                    Ex3Fluct::Random => "-random",
                },
            ),
        )?;
    }

    // Example 4.
    for with_fluct in [true, false].iter().copied() {
        for (equilibrium, name) in
            Equilibrium::iter_all().zip(["neg", "zero", "pos"].iter().copied())
        {
            println!(
                "[Example 4] Running simulation with{} fluctuations and with equilibrium kind: {:?}",
                if with_fluct { "" } else { "out" },
                equilibrium,
            );
            run_simulation(
                sim_params_ex4.clone(),
                x0_ex4.clone(),
                &mut |t| {
                    if with_fluct {
                        DVector::repeat(1, 1.0 / (1.0 + 2.0 * t))
                    } else {
                        DVector::zeros(1)
                    }
                },
                &mut ProposedController::new(
                    ControllerParams {
                        equilibrium,
                        ..ctl_params_ex4a.clone()
                    },
                    &mut rng,
                    SineTauFn {
                        mu: 0.3,
                        tau: 1.3,
                        phase: 0.0,
                    },
                ),
                100000,
                10000,
                0.1,
                0.0,
                &format!(
                    "data-out/ex4/a-with{}-fluct-eq-{}.csv",
                    if with_fluct { "" } else { "out" },
                    name,
                ),
            )?;
            run_simulation(
                sim_params_ex4.clone(),
                x0_ex4.clone(),
                &mut |t| {
                    if with_fluct {
                        DVector::repeat(1, 1.0 / (1.0 + 2.0 * t))
                    } else {
                        DVector::zeros(1)
                    }
                },
                &mut ProposedController::new(
                    ControllerParams {
                        equilibrium,
                        ..ctl_params_ex4b.clone()
                    },
                    &mut rng,
                    SineTauFn {
                        mu: 0.3,
                        tau: 1.3,
                        phase: 0.0,
                    },
                ),
                100000,
                10000,
                0.1,
                0.0,
                &format!(
                    "data-out/ex4/b-with{}-fluct-eq-{}.csv",
                    if with_fluct { "" } else { "out" },
                    name,
                ),
            )?;
        }
    }

    Ok(())
}

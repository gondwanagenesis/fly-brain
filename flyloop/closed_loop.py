"""CLOSED SENSORIMOTOR LOOP: connectome brain <-> physical body in a virtual world.

    world (MuJoCo/NeuroMechFly)
        |  sugar within reach of the proboscis?
        v
    GATE 1  inject Poisson drive at the 21 sugar GRN flywire ids
        |
    139k-neuron FlyWire LIF brain  (BrainEngine, event-driven)
        |
    GATE 2  read firing rate of the brain's motor neurons (super_class == 'motor')
        v
    proboscis (rostrum + haustellum) extends -> world changes -> repeat

Brain runs at dt=0.1 ms, physics at dt=0.1 ms, they handshake every EXCHANGE_MS.
Nothing here is trained: the behaviour comes from connectome wiring alone.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, mujoco, imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flyloop"))
from brain_engine import BrainEngine, RateTracker, DT

from flygym import Simulation
from flygym.compose import (NeuroMechFly, FlatGroundWorld, ActuatorType,
                            KinematicPosePreset)
from flygym.anatomy import ContactBodiesPreset
from flygym.utils.math import Rotation3D

SUGAR_GRNS = [720575940624963786,720575940630233916,720575940637568838,
720575940638202345,720575940617000768,720575940630797113,720575940632889389,
720575940621754367,720575940621502051,720575940640649691,720575940639332736,
720575940616885538,720575940639198653,720575940639259967,720575940617937543,
720575940632425919,720575940633143833,720575940612670570,720575940628853239,
720575940629176663,720575940611875570]

EXCHANGE_MS = 15.0      # brain<->body handshake period (as in Eon's loop)
STIM_HZ = 200.0         # GRN drive when the fly tastes sugar
SUGAR_ON_MS = 150.0     # sugar presented at t=150 ms


def motor_output_ids(data_dir):
    """GATE 2 population: the brain's own motor neurons (head/proboscis)."""
    ann = pd.read_csv(Path(data_dir) / "flywire_meta" / "neuron_annotations.tsv",
                      sep="\t", low_memory=False)
    m = ann[ann.super_class == "motor"]
    return [int(x) for x in m.root_id.dropna()]


def build_body():
    fly = NeuroMechFly()
    try: fly.colorize()
    except Exception: pass
    skel = fly._get_base_skeleton()
    jd = fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.add_actuators(list(jd.keys()), ActuatorType.POSITION,
                      neutral_input=KinematicPosePreset.NEUTRAL)
    fly.add_leg_adhesion(gain=60.0)
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=np.array([0, 0, 2.0]),
                  spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY)
    sim = Simulation(world, timestep=1e-4)
    sim.reset()
    return fly, sim


def proboscis_actuators(fly, mjm):
    """Indices (within the POSITION actuator vector) of the proboscis joints."""
    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    idx = {}
    for k, dof in enumerate(order):
        ch, ax = dof.child.name, str(dof.axis).split(".")[-1]
        if ch in ("c_rostrum", "c_haustellum") and "PITCH" in ax:
            idx[ch] = k
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-ms", type=float, default=400.0)
    ap.add_argument("--out", default=str(ROOT / "virtualfly" / "closed_loop.mp4"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()

    print("building body ...")
    fly, sim = build_body()
    mjm, mjd = sim.mj_model, sim.mj_data
    pos_idx = [i for i in range(mjm.nu)
               if mjm.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT]
    sim.warmup(0.05)
    base_ctrl = mjd.ctrl[pos_idx].copy()
    prob = proboscis_actuators(fly, mjm)
    print(f"  proboscis actuators: {prob}")

    print("loading brain ...")
    eng = BrainEngine(data_dir=a.data, stim_ids=SUGAR_GRNS, seed=0)
    out_ids = motor_output_ids(a.data)
    out_idx = eng.indices_of(out_ids)
    tracker = RateTracker(eng, out_idx, tau_ms=40.0)
    print(f"  GATE1 sugar GRNs: {len(SUGAR_GRNS)}   "
          f"GATE2 motor neurons: {len(out_idx)}")

    tid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "nmf/c_thorax")
    r = mujoco.Renderer(mjm, height=544, width=800)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 5.0; cam.elevation = -12; cam.azimuth = 152

    brain_steps = int(EXCHANGE_MS / DT)          # 150 brain steps per exchange
    phys_steps = int(EXCHANGE_MS / 1000 / 1e-4)  # 150 physics steps per exchange
    n_exchange = int(a.duration_ms / EXCHANGE_MS)
    frames, log = [], []
    ext = 0.0

    print(f"running {n_exchange} exchanges "
          f"({a.duration_ms} ms, handshake every {EXCHANGE_MS} ms) ...")
    for k in range(n_exchange):
        t_ms = k * EXCHANGE_MS
        # ---- GATE 1: world -> brain -------------------------------------
        tasting = t_ms >= SUGAR_ON_MS
        eng.inject(STIM_HZ if tasting else 0.0)

        # ---- brain ------------------------------------------------------
        nsp = 0
        for _ in range(brain_steps):
            nsp += int(eng.step().sum().item())
            tracker.update()

        # ---- GATE 2: brain -> world -------------------------------------
        mrate = float(tracker.r.mean())          # Hz, motor-neuron population
        drive = np.clip(mrate / 12.0, 0.0, 1.0)  # -> normalized motor command
        ext += 0.25 * (drive - ext)              # smooth muscle response

        ctrl = base_ctrl.copy()
        if "c_rostrum" in prob:    ctrl[prob["c_rostrum"]]    += 1.2 * ext
        if "c_haustellum" in prob: ctrl[prob["c_haustellum"]] += 1.0 * ext
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, ctrl)

        # ---- world ------------------------------------------------------
        for _ in range(phys_steps):
            sim.step()

        log.append(dict(t_ms=t_ms, tasting=tasting, spikes=nsp,
                        motor_hz=mrate, extension=ext))
        if not a.headless:
            cam.lookat[:] = mjd.xpos[tid]
            r.update_scene(mjd, camera=cam)
            frames.append(r.render())
        print(f"  t={t_ms:6.0f}ms  sugar={'ON ' if tasting else 'off'}  "
              f"spikes={nsp:5d}  motor={mrate:6.2f}Hz  proboscis={ext:.3f}")

    df = pd.DataFrame(log)
    csv = Path(a.out).with_suffix(".csv")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    if frames:
        imageio.mimwrite(a.out, frames, fps=12, quality=8)
        imageio.imwrite(str(Path(a.out).with_name("loop_before.png")),
                        frames[max(0, int(SUGAR_ON_MS/EXCHANGE_MS) - 2)])
        imageio.imwrite(str(Path(a.out).with_name("loop_after.png")), frames[-1])
    pre = df[~df.tasting].extension.max() if (~df.tasting).any() else 0
    post = df.extension.max()
    print(f"\nproboscis extension  before sugar: {pre:.3f}   after: {post:.3f}")
    print(f"saved {a.out} + {csv}")


if __name__ == "__main__":
    main()

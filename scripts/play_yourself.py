"""
play_beer_game.py -- play the Beer Game by hand to sanity-check the environment.

Standard library only (tkinter ships with CPython). Uses PyYAML if available
(it is, in any Hydra venv) and otherwise a tiny built-in parser, so it runs with
no extra installs either way.

READS YOUR ACTUAL CONFIG. The env settings come from conf/config.yaml's `env:`
block -- the same file Hydra uses -- so anything you change there shows up here.
The loaded file path and the resulting env settings are printed in the window
header, so you can confirm your edits took effect.

  (Hydra's `defaults:`/agent composition is NOT resolved -- this tool only needs
   the plain `env:` section, which is ordinary YAML. The agent block is irrelevant
   to playing the physics by hand.)

RUN
  From your project root (so conf/config.yaml is found):
      python scripts/play_beer_game.py
  or point it explicitly:
      python scripts/play_beer_game.py --config conf/config.yaml
"""
import os
import sys
import argparse
import tkinter as tk
from tkinter import ttk, messagebox

# --- import the env (edit if your layout differs) -----------------------------
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
try:
    from beer_game_env import BeerGameParallelEnv
except Exception:
    try:
        from envs.beer_game_env import BeerGameParallelEnv
    except Exception as e:
        raise SystemExit(
            "Could not import BeerGameParallelEnv. Run from the project root or put "
            "this file where it can import the env. Original error: %r" % e
        )

SEED = 42


# ==============================================================================
# Config loading: read the `env:` block from conf/config.yaml.
# ==============================================================================
def _find_config(explicit=None):
    """Locate config.yaml. Try, in order: --config, ./conf/config.yaml relative to
    cwd, ../conf/config.yaml relative to this file, and a few common spots."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        os.path.join(os.getcwd(), "conf", "config.yaml"),
        os.path.join(os.path.dirname(_here), "conf", "config.yaml"),
        os.path.join(_here, "conf", "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit(
        "Could not find config.yaml. Looked in:\n  " + "\n  ".join(candidates) +
        "\nPass it explicitly:  python scripts/play_beer_game.py --config conf/config.yaml"
    )


def _coerce(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_coerce(x) for x in inner.split(",")] if inner else []
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _parse_env_block_stdlib(text):
    """Minimal fallback: extract the top-level `env:` mapping as flat key/values.
    Handles ints, floats, bools, quoted strings, and [a, b] lists -- enough for the
    env config. Not a general YAML parser."""
    env = {}
    in_env = False
    env_indent = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.rstrip()
        indent = len(stripped) - len(stripped.lstrip())
        key_part = stripped.strip()
        if not in_env:
            if key_part.rstrip(":") == "env" and key_part.endswith(":"):
                in_env = True
                env_indent = indent
            continue
        if indent <= env_indent:
            break
        if ":" in key_part:
            k, _, v = key_part.partition(":")
            v = v.split("#")[0].strip()
            if v == "":
                continue
            env[k.strip()] = _coerce(v)
    return env


def load_env_config(path):
    """Return (env_dict, source_label). Prefer PyYAML; fall back to the stdlib parser."""
    with open(path, "r") as f:
        text = f.read()
    try:
        import yaml  # PyYAML is present in any Hydra install
        full = yaml.safe_load(text) or {}
        env = full.get("env", {})
        if not isinstance(env, dict) or not env:
            raise ValueError("no `env:` mapping found")
        return dict(env), "PyYAML"
    except ImportError:
        env = _parse_env_block_stdlib(text)
        if not env:
            raise SystemExit(f"Parsed no `env:` keys from {path} (stdlib fallback).")
        return env, "stdlib-fallback"


def _cfg_lead_summary(cfg):
    def one(fixed_key, range_key, default):
        if cfg.get(range_key) is not None:
            lo, hi = cfg[range_key]; return f"U[{lo},{hi}]"
        if cfg.get(fixed_key) is not None:
            return str(cfg[fixed_key])
        return f"{default}(default)"
    legacy = cfg.get("jittery_lead_time", False) and "ship_lead_time" not in cfg and "ship_lead_time_range" not in cfg
    ship = "U[1,9] (legacy jitter)" if legacy else one("ship_lead_time", "ship_lead_time_range", 2)
    return (f"order={one('order_lead_time','order_lead_time_range',2)}  "
            f"order_mfr={one('order_lead_time_mfr','order_lead_time_mfr_range',1)}  "
            f"ship={ship}  "
            f"production={one('production_lead_time','production_lead_time_range',2)}")


class BeerGameGUI:
    def __init__(self, root, env_config, source_label, config_path):
        self.root = root
        root.title("Beer Game — manual play / correctness check")
        self.config_path = config_path
        self.source_label = source_label

        self.env = BeerGameParallelEnv(env_config)

        # everything derived from the constructed env, nothing hardcoded
        self.agents = list(self.env.possible_agents)
        self.max_order = self.env.max_order
        # the agent obs no longer contains the pipeline; we still SHOW it as a physics
        # debug view. Depth = the env's max possible lead time (not agent-visible).
        self.pipe_window = int(getattr(self.env, "_max_lead", 4))
        self.horizon = self.env.horizon
        self.h = self.env.h
        self.b = self.env.b
        self.lead_summary = _cfg_lead_summary(self.env.config)
        self.obs_pipe_labels = [f"+{t}" for t in range(1, self.pipe_window + 1)]

        self.obs, _ = self.env.reset(seed=SEED)
        self.done = False
        self.cum_cost = 0.0
        self.last_step_cost = 0.0
        self.default_order = self._infer_default_order()

        self._build_widgets()
        self._refresh()

    def _infer_default_order(self):
        try:
            sp = self.env.shipment_pipelines[self.agents[0]].pipeline
            hint = max(sp.values()) if sp else 0
            return int(hint) if hint > 0 else 1
        except Exception:
            return 1

    def _build_widgets(self):
        pad = {"padx": 6, "pady": 3}

        top = ttk.Frame(self.root); top.grid(row=0, column=0, sticky="nw", **pad)
        self.week_var = tk.StringVar()
        ttk.Label(top, textvariable=self.week_var, font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=f"config: {self.config_path}  [{self.source_label}]",
                  font=("TkDefaultFont", 8)).grid(row=1, column=0, sticky="w")
        ttk.Label(top, text=f"demand_type={self.env.config.get('demand_type')}   "
                            f"max_order={self.max_order}   h={self.h}  b={self.b}",
                  font=("TkDefaultFont", 9)).grid(row=2, column=0, sticky="w")
        ttk.Label(top, text=f"lead times:  {self.lead_summary}",
                  font=("TkDefaultFont", 9)).grid(row=3, column=0, sticky="w")

        pipe_hdr = "in-transit(physics,not obs) [" + ",".join(self.obs_pipe_labels) + "]"
        cols = ["echelon", "inventory", "backlog", "on_order", "last_demand", pipe_hdr, "your order"]
        hdr = ttk.Frame(self.root); hdr.grid(row=1, column=0, sticky="nw", **pad)
        for c, name in enumerate(cols):
            ttk.Label(hdr, text=name, font=("TkDefaultFont", 9, "bold"),
                      width=max(12, len(name) + 1), anchor="w").grid(row=0, column=c, sticky="w")

        self.rows = {}
        self.order_entries = {}
        for r, a in enumerate(self.agents, start=1):
            ttk.Label(hdr, text=a, width=12, anchor="w").grid(row=r, column=0, sticky="w")
            labels = {}
            for c, key in enumerate(["inv", "back", "onord", "lastd", "pipe"], start=1):
                w = max(12, len(pipe_hdr) + 1) if key == "pipe" else 12
                lab = ttk.Label(hdr, text="-", width=w, anchor="w")
                lab.grid(row=r, column=c, sticky="w"); labels[key] = lab
            ent = ttk.Entry(hdr, width=8); ent.insert(0, str(self.default_order))
            ent.grid(row=r, column=6, sticky="w")
            self.rows[a] = labels
            self.order_entries[a] = ent

        cost = ttk.Frame(self.root); cost.grid(row=2, column=0, sticky="nw", **pad)
        self.cost_var = tk.StringVar()
        ttk.Label(cost, textvariable=self.cost_var, font=("TkDefaultFont", 10)).grid(row=0, column=0, sticky="w")

        btns = ttk.Frame(self.root); btns.grid(row=3, column=0, sticky="nw", **pad)
        self.step_btn = ttk.Button(btns, text="Step week  \u25b6", command=self.step)
        self.step_btn.grid(row=0, column=0, padx=3)
        ttk.Button(btns, text=f"Fill all = {self.default_order}",
                   command=lambda: self._fill_all(self.default_order)).grid(row=0, column=1, padx=3)
        ttk.Button(btns, text="Fill all = 0", command=lambda: self._fill_all(0)).grid(row=0, column=2, padx=3)
        ttk.Button(btns, text="Reset", command=self.reset).grid(row=0, column=3, padx=3)
        ttk.Button(btns, text="Reload config", command=self.reload_config).grid(row=0, column=4, padx=3)

        audit = ttk.Frame(self.root); audit.grid(row=4, column=0, sticky="nw", **pad)
        ttk.Label(audit, text="OBSERVATION each agent actually sees (4 scalars): "
                              "[inv, backlog, on_order, last_demand]   "
                              "-- the in-transit pipeline is NOT in the obs:",
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.obs_text = tk.Text(audit, width=96, height=len(self.agents) + 3, font=("Courier", 9))
        self.obs_text.grid(row=1, column=0, sticky="w")

        log = ttk.Frame(self.root); log.grid(row=5, column=0, sticky="nw", **pad)
        ttk.Label(log, text="History (week | orders | step cost | realised retailer demand):",
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(log, width=96, height=8, font=("Courier", 9))
        self.log_text.grid(row=1, column=0, sticky="w")

    def _fill_all(self, v):
        for a in self.agents:
            self.order_entries[a].delete(0, tk.END); self.order_entries[a].insert(0, str(v))

    def _pipe_vals(self, agent):
        sp = self.env.shipment_pipelines[agent].pipeline
        cs = self.env.current_step
        return [int(sp.get(cs + t, 0)) for t in range(1, self.pipe_window + 1)]

    def _max_ship_slot(self):
        cfg = self.env.config
        if cfg.get("ship_lead_time_range") is not None:
            return min(self.pipe_window, int(cfg["ship_lead_time_range"][1]))
        if cfg.get("jittery_lead_time", False) and "ship_lead_time" not in cfg:
            return min(self.pipe_window, 9)
        return min(self.pipe_window, max(int(cfg.get("ship_lead_time", 2)), 2))

    def _refresh(self):
        self.week_var.set(f"Week {self.env.current_step} / {self.horizon}"
                          + ("   \u2014  EPISODE DONE" if self.done else ""))
        for a in self.agents:
            L = self.rows[a]
            L["inv"].config(text=f"{self.env.inventory[a]:.0f}")
            L["back"].config(text=f"{self.env.backlog[a]:.0f}")
            L["onord"].config(text=f"{self.env.unfulfilled_orders[a]:.0f}")
            L["lastd"].config(text=f"{self.env.current_incoming_order[a]:.0f}")
            L["pipe"].config(text="[" + ",".join(str(v) for v in self._pipe_vals(a)) + "]")
        self.cost_var.set(f"Last week cost: {self.last_step_cost:8.2f}     "
                          f"Cumulative cost: {self.cum_cost:10.2f}")

        self.obs_text.delete("1.0", tk.END)
        for a in self.agents:
            o = self.obs[a]
            self.obs_text.insert(tk.END, f"{a:12s} " + " ".join(f"{x:6.1f}" for x in o) + "\n")
        live = self._max_ship_slot()
        empties = self.pipe_window - live
        if empties > 0:
            self.obs_text.insert(tk.END,
                f"\n(note: shipping lead time fills at most pipe slots +1..+{live}; the last "
                f"{empties} slot(s) stay 0 until a wider ship lead time is configured.)")

        if self.done:
            self.step_btn.config(state="disabled")

    def step(self):
        if self.done:
            return
        try:
            orders = {}
            for a in self.agents:
                v = int(float(self.order_entries[a].get()))
                if v < 0 or v > self.max_order:
                    raise ValueError(f"{a}: order must be 0..{self.max_order}")
                orders[a] = v
        except ValueError as e:
            messagebox.showerror("Invalid order", str(e)); return

        acts = {a: [orders[a] / self.max_order] for a in self.agents}
        self.obs, rew, term, trunc, infos = self.env.step(acts)
        self.last_step_cost = sum(infos[a]["local_cost"] for a in self.agents)
        self.cum_cost += self.last_step_cost
        self.done = any(trunc.values()) or any(term.values())

        ostr = ",".join(str(orders[a]) for a in self.agents)
        retailer = self.agents[0]
        # realized retailer demand is nested under training_targets (the env keeps the obs clean);
        # .get() chain so this never KeyErrors if the env's info schema changes.
        realized_demand = infos[retailer].get("training_targets", {}).get("demand", 0.0)
        self.log_text.insert(tk.END,
            f"wk {self.env.current_step:2d} | ord {ostr:>16s} | cost {self.last_step_cost:8.2f} "
            f"| demand({retailer})={realized_demand:.0f}\n")
        self.log_text.see(tk.END)
        self._refresh()

    def reset(self):
        self.obs, _ = self.env.reset(seed=SEED)
        self.done = False; self.cum_cost = 0.0; self.last_step_cost = 0.0
        self.step_btn.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self._refresh()

    def reload_config(self):
        """Re-read config.yaml and rebuild the env without restarting the app."""
        try:
            env_config, source = load_env_config(self.config_path)
            new_env = BeerGameParallelEnv(env_config)
        except Exception as e:
            messagebox.showerror("Reload failed", str(e)); return
        # tear down and rebuild the whole UI so widths/agents/lookahead update
        for child in list(self.root.children.values()):
            child.destroy()
        self.__init__(self.root, env_config, source, self.config_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to config.yaml (default: auto-find conf/config.yaml)")
    args = ap.parse_args()

    path = _find_config(args.config)
    env_config, source = load_env_config(path)
    print(f"[play] loaded env config from {path}  ({source})")
    print(f"[play] env settings: {env_config}")

    root = tk.Tk()
    BeerGameGUI(root, env_config, source, path)
    root.mainloop()


if __name__ == "__main__":
    main()